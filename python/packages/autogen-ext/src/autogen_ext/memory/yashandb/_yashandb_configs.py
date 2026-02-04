"""Configuration classes for YashanDB vector memory."""

from typing import Any, Callable, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ._yashandb_types import YashanDBDistanceMetric


class DefaultEmbeddingFunctionConfig(BaseModel):
    """Configuration for the default YashanDB embedding function.

    Uses YashanDB's default embedding function (Sentence Transformers all-MiniLM-L6-v2).

    .. versionadded:: v0.7.5
       Support for custom embedding functions in YashanDB memory.
    """

    function_type: Literal["default"] = "default"


class SentenceTransformerEmbeddingFunctionConfig(BaseModel):
    """Configuration for SentenceTransformer embedding functions.

    Allows specifying a custom SentenceTransformer model for embeddings.

    .. versionadded:: v0.7.5
       Support for custom embedding functions in YashanDB memory.

    Args:
        model_name (str): Name of the SentenceTransformer model to use.
            Defaults to "all-MiniLM-L6-v2".

    Example:
        .. code-block:: python

            from autogen_ext.memory.yashandb import SentenceTransformerEmbeddingFunctionConfig

            _ = SentenceTransformerEmbeddingFunctionConfig(model_name="paraphrase-multilingual-mpnet-base-v2")
    """

    function_type: Literal["sentence_transformer"] = "sentence_transformer"
    model_name: str = Field(default="all-MiniLM-L6-v2", description="SentenceTransformer model name to use")


# Tagged union type for embedding function configurations
EmbeddingFunctionConfig = Annotated[
    Union[DefaultEmbeddingFunctionConfig, SentenceTransformerEmbeddingFunctionConfig],
    Field(discriminator="function_type"),
]


class YashanDBVectorMemoryConfig(BaseModel):
    """Base configuration for YashanDB-based memory implementation.

    .. versionadded:: v0.7.5
       Added support for custom embedding functions via embedding_function_config.
    """

    # Connection-related fields
    host: str = Field(default="127.0.0.1", description="Host of the remote server")
    port: int = Field(default=1688, description="Port of the remote server")
    user: str = Field(default="yashan_user", description="Database user of the remote server")
    password: str = Field(default="yashan_password", description="Password of the remote server")

    collection_name: str = Field(default="memory_store", description="Name of the YashanDB collection")
    distance_metric: YashanDBDistanceMetric = Field(
        default="cosine",
        description="Distance metric for similarity search",
    )
    k: int = Field(default=3, description="Number of results to return in queries")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity score threshold")

    m: int = Field(default=16, description="The M value for HNSW index")
    ef: int = Field(default=100, description="The ef construction value for HNSW index")

    embedding_function_config: EmbeddingFunctionConfig = Field(
        default_factory=DefaultEmbeddingFunctionConfig,
        description="Configuration for the embedding function",
    )
