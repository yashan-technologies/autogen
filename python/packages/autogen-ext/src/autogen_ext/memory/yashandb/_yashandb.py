import asyncio
import functools
import logging
import uuid
from typing import Any, Callable, List, Optional, Tuple

from autogen_core import CancellationToken, Component, Image
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from typing_extensions import Self
from yasdb.libs.exceptions import DatabaseError, InterfaceError

from ._yashandb_client import YashanDBClient, YashanDBCollection
from ._yashandb_configs import YashanDBVectorMemoryConfig
from ._yashandb_embeddings import create_embedding_function_with_dimension
from ._yashandb_types import ID, Document, EmbeddingFunction, Metadata, T

logger = logging.getLogger(__name__)


def _retry_connections(retries: int = 3, sleep: float = 0.5) -> Callable[[Callable], Callable]:
    def decorator(method):
        @functools.wraps(method)
        async def new_method(self, *args, **kwargs):
            try:
                return await method(self, *args, **kwargs)
            except DatabaseError as e:
                # YAS-00406 connection is closed
                # YAS-08012 connection has been disconnected
                if all(code not in str(e) for code in ("YAS-00406", "YAS-08012")):
                    raise
            except InterfaceError as e:
                if str(e) != "not connected":
                    raise

            # allow retry connection
            nonlocal retries
            logger.warning(f"Connection disconnected, will retry {retries} times.")
            while retries:
                retries -= 1
                try:
                    logger.info(f"Retrying connection, {retries} retries remaining...")
                    self._connect()
                except DatabaseError as e:
                    # YAS-00402 failed to connect socket
                    if "YAS-00402" not in str(e):
                        raise

                    # sleep to avoid congesting network with retry requests
                    await asyncio.sleep(sleep)
                    continue
                break

            logger.info("Reconnected.")
            return await method(self, *args, **kwargs)

        return new_method

    return decorator


class YashanDBVectorMemory(Memory, Component[YashanDBVectorMemoryConfig]):
    """
    Store and retrieve memory using vector similarity search powered by YashanDB.

    `YashanDBVectorMemory` provides a vector-based memory implementation that uses YashanDB for
    storing and retrieving content based on semantic similarity. It enhances agents with the ability
    to recall contextually relevant information during conversations by leveraging vector embeddings
    to find similar content.

    This implementation serves as a reference for more complex memory systems using vector embeddings.
    For advanced use cases requiring specialized formatting of retrieved content, users should extend
    this class and override the `update_context()` method.

    Args:
        config (YashanDBVectorMemoryConfig | None): Configuration for the YashanDB memory.
            If None, defaults to a YashanDBVectorMemoryConfig with default values.

    Example:

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.ui import Console
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.yashandb import (
                SentenceTransformerEmbeddingFunctionConfig,
                YashanDBVectorMemory,
                YashanDBVectorMemoryConfig,
            )
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            def get_weather(city: str) -> str:
                return f"The weather in {city} is sunny with a high of 90°F and a low of 70°F."


            def fahrenheit_to_celsius(fahrenheit: float) -> float:
                return (fahrenheit - 32) * 5.0 / 9.0


            async def main() -> None:
                host: str = "127.0.0.1"
                port: int = 1688
                user: str = "yashan_user"
                password: str = "yashan_password"

                # Use default embedding function
                default_memory = YashanDBVectorMemory(
                    config=YashanDBVectorMemoryConfig(
                        host=host,
                        port=port,
                        user=user,
                        password=password,
                        collection_name="user_preferences",
                        k=3,  # Return top 3 results
                        score_threshold=0.5,  # Minimum similarity score
                    )
                )

                # Using a custom SentenceTransformer model
                custom_memory = YashanDBVectorMemory(
                    config=YashanDBVectorMemoryConfig(
                        host=host,
                        port=port,
                        user=user,
                        password=password,
                        collection_name="multilingual_memory",
                        embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                            model_name="paraphrase-multilingual-mpnet-base-v2"
                        ),
                    )
                )

                # Add user preferences to memory
                await custom_memory.add(
                    MemoryContent(
                        content="The user prefers weather temperatures in Celsius",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"category": "preferences", "type": "units"},
                    )
                )

                ## needed for self-hosted openai models
                model_info = ModelInfo(
                    vision=False,
                    function_calling=True,
                    json_output=False,
                    family="unknown",
                    structured_output=False,
                    multiple_system_messages=True,
                )
                _ = model_info

                # Create assistant agent with YashanDB memory
                assistant = AssistantAgent(
                    name="assistant",
                    model_client=OpenAIChatCompletionClient(
                        model="gpt-4.1",
                        # base_url="...",
                        # api_key="...",
                        # model_info=model_info,  # needed for self-hosted models
                    ),
                    tools=[
                        get_weather,
                        fahrenheit_to_celsius,
                    ],
                    max_tool_iterations=10,
                    memory=[custom_memory],
                )

                # The memory will automatically retrieve relevant content during conversations
                await Console(assistant.run_stream(task="What's the temperature in New York?"))

                # Remember to close the memory when finished
                await default_memory.close()
                await custom_memory.close()


            asyncio.run(main())


        Output:

        .. code-block:: text

            ---------- TextMessage (user) ----------
            What's the temperature in New York?
            ---------- MemoryQueryEvent (assistant) ----------
            [MemoryContent(content='The user prefers weather temperatures in Celsius', mime_type='MemoryMimeType.TEXT', metadata={'type': 'units', 'category': 'preferences', 'mime_type': 'MemoryMimeType.TEXT', 'score': 0.7629481554031372, 'id': '40e84e2b-bc8d-4ce2-bdd5-4cee5b532614'}), MemoryContent(content='The user prefers weather temperatures in Celsius', mime_type='MemoryMimeType.TEXT', metadata={'type': 'units', 'category': 'preferences', 'mime_type': 'MemoryMimeType.TEXT', 'score': 0.7629481554031372, 'id': '58630265-6129-4bd9-85a1-5c14ca524a22'}), MemoryContent(content='The user prefers weather temperatures in Celsius', mime_type='MemoryMimeType.TEXT', metadata={'type': 'units', 'category': 'preferences', 'mime_type': 'MemoryMimeType.TEXT', 'score': 0.7629481554031372, 'id': '82933eeb-ae23-4206-9876-f5d2b8464ecc'})]
            ---------- ThoughtEvent (assistant) ----------



            ---------- ToolCallRequestEvent (assistant) ----------
            [FunctionCall(id='chatcmpl-tool-6d7d6973d3794f36aae47158830518e5', arguments='{"city": "New York"}', name='get_weather')]
            ---------- ToolCallExecutionEvent (assistant) ----------
            [FunctionExecutionResult(content='The weather in New York is sunny with a high of 90°F and a low of 70°F.', name='get_weather', call_id='chatcmpl-tool-6d7d6973d3794f36aae47158830518e5', is_error=False)]
            ---------- ThoughtEvent (assistant) ----------

            Okay, the user asked for the temperature in New York. I called the get_weather function with the city parameter set to "New York". The response came back with the weather details: sunny, high of 90°F, and low of 70°F. Now, I need to present this information in a clear and friendly way. Since the user specified they prefer Celsius temperatures, I should convert the Fahrenheit to Celsius. Let me check the conversion from 90°F to Celsius. The formula is (F - 32) * 5/9. Plugging
            in 90, that's (90-32)*5/9 = 58*5/9 ≈ 32.2°C. Similarly, 70°F is (70-32)*5/9 = 38*5/9 ≈ 21.1°C. So the correct temperatures in Celsius are approximately 32.2°C and 21.1°C. I should mention both temperatures in Celsius to match the user's preference. Let me structure the answer to include both values clearly.

            ---------- TextMessage (assistant) ----------


            The temperature in New York is currently 32.2°C (90°F) on the high and 21.1°C (70°F) on the low.

    """

    component_config_schema = YashanDBVectorMemoryConfig
    component_provider_override = "autogen_ext.memory.yashandb.YashanDBVectorMemory"

    _config: YashanDBVectorMemoryConfig
    _client: YashanDBClient
    _collection: YashanDBCollection

    _embed: EmbeddingFunction
    _dim: int

    def __init__(self, config: Optional[YashanDBVectorMemoryConfig] = None) -> None:
        self._config = config or YashanDBVectorMemoryConfig()
        self._embed, self._dim = self._create_embedding_function()
        self._connect()

    @property
    def collection_name(self) -> str:
        """Get the name of the YashanDB collection."""
        return self._config.collection_name

    def _create_embedding_function(self) -> Tuple[EmbeddingFunction, int]:
        """Create an embedding function based on the configuration.

        Returns:
            [0]: A YashanDB-compatible embedding function.
            [1]: The dimension of the embedding function.

        Raises:
            ValueError: If the embedding function type is unsupported.
            ImportError: If required dependencies are not installed.
        """

        return create_embedding_function_with_dimension(self._config.embedding_function_config)

    def _connect(self):
        self._client = YashanDBClient(self._config, self._embed, self._dim)
        try:
            self._collection = self._client.get_or_create_collection(
                name=self._config.collection_name,
                vector_dimension=self._dim,
                embedding_function=self._embed,
                distance_metric=self._config.distance_metric,
            )

        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise

    def _extract_text(self, content_item: str | MemoryContent) -> str:
        """Extract searchable text from content."""
        if isinstance(content_item, str):
            return content_item

        content = content_item.content
        mime_type = content_item.mime_type

        if mime_type in [MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN]:
            return str(content)
        elif mime_type == MemoryMimeType.JSON:
            if isinstance(content, dict):
                # Store original JSON string representation
                return str(content).lower()
            raise ValueError("JSON content must be a dict")
        elif isinstance(content, Image):
            raise ValueError("Image content cannot be converted to text")
        else:
            raise ValueError(f"Unsupported content type: {mime_type}")

    def _calculate_score(self, distance: float) -> float:
        """Convert YashanDB distance to a similarity score."""
        if self._config.distance_metric == "cosine":
            return 1.0 - (distance / 2.0)
        return 1.0 / (1.0 + distance)

    @_retry_connections()
    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        # Extract query from last message
        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)

        # Query memory and get results
        query_results = await self.query(query_text)

        if query_results.results:
            # Format results for context
            memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(query_results.results, 1)]
            memory_context = "\nRelevant memory content:\n" + "\n".join(memory_strings)

            # Add to context
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=query_results)

    @_retry_connections()
    async def add(self, content: MemoryContent, cancellation_token: Optional[CancellationToken] = None) -> None:
        try:
            # Extract text from content
            text = self._extract_text(content)

            # Use metadata directly from content
            metadata_dict = content.metadata or {}
            metadata_dict["mime_type"] = str(content.mime_type)

            # Add to YashanDB
            self._collection.add(document=text, metadata=metadata_dict, id=str(uuid.uuid4()))

        except Exception as e:
            logger.error(f"Failed to add content to YashanDB: {e}")
            raise

    @_retry_connections()
    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        try:
            # Extract text for query
            query_text = self._extract_text(query)

            # Query YashanDB
            results = self._collection.query(
                query_texts=[query_text],
                n_results=self._config.k,
                **kwargs,
            )

            # Convert results to MemoryContent list
            memory_results: List[MemoryContent] = []

            if not (results and results.documents and results.metadatas and results.distances):
                return MemoryQueryResult(results=memory_results)

            def get_or_default(lst: List[T], idx: int, default: T) -> T:
                return lst[idx] if len(lst) > idx else default

            documents: List[Document] = get_or_default(results.documents, 0, [])
            metadatas: List[Metadata] = get_or_default(results.metadatas, 0, [])
            distances: List[float] = get_or_default(results.distances, 0, [])
            ids: List[ID] = get_or_default(results.ids or [], 0, [])

            for doc, metadata_dict, distance, doc_id in zip(documents, metadatas, distances, ids, strict=False):
                # Calculate score
                score = self._calculate_score(distance)
                metadata = dict(metadata_dict)
                metadata["score"] = score
                metadata["id"] = doc_id
                if self._config.score_threshold is not None and score < self._config.score_threshold:
                    continue

                # Extract mime_type from metadata
                mime_type = str(metadata_dict.get("mime_type", MemoryMimeType.TEXT.value))

                # Create MemoryContent
                content = MemoryContent(
                    content=doc,
                    mime_type=mime_type,
                    metadata=metadata,
                )
                memory_results.append(content)

            return MemoryQueryResult(results=memory_results)

        except Exception as e:
            logger.error(f"Failed to query YashanDB: {e}")
            raise

    @_retry_connections()
    async def clear(self) -> None:
        try:
            self._collection.clear()
        except Exception as e:
            logger.error(f"Failed to clear YashanDB collection: {e}")
            raise

    async def close(self) -> None:
        """Clean up YashanDB client and resources."""
        self._client.close()

    async def reset(self) -> None:
        raise NotImplementedError("Reset not implemented.")

    def _to_config(self) -> YashanDBVectorMemoryConfig:
        """Serialize the memory configuration."""
        return self._config

    @classmethod
    def _from_config(cls, config: YashanDBVectorMemoryConfig) -> Self:
        """Deserialize the memory configuration."""
        return cls(config=config)
