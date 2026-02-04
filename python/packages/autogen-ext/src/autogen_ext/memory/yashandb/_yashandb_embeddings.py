import logging
from array import array
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, cast

from typing_extensions import Self

from ._yashandb_configs import (
    DefaultEmbeddingFunctionConfig,
    EmbeddingFunctionConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)
from ._yashandb_types import Document, Embedding, EmbeddingFunction

if TYPE_CHECKING:
    try:
        from sentence_transformers import SentenceTransformer
    except:  # noqa: E722
        pass


def create_embedding_function(cfg: EmbeddingFunctionConfig):
    if isinstance(cfg, DefaultEmbeddingFunctionConfig):
        return _default_embedding_function(cfg)

    if isinstance(cfg, SentenceTransformerEmbeddingFunctionConfig):
        return _sentence_transformer_function(cfg)

    raise NotImplementedError


def _resolve_sentence_transformer_dim(emb: EmbeddingFunction) -> Optional[int]:
    if not isinstance(emb, SentenceTransformerEmbeddingFunction):
        return None

    model = emb._model
    try:
        dim = model.get_sentence_embedding_dimension()
    except Exception:
        return None

    if dim is None or dim <= 0:
        return None

    # Embedding model may be truncated
    truncate_dim = model.truncate_dim
    if truncate_dim is not None and truncate_dim < dim:
        dim = truncate_dim

    return dim


def _get_embedding_function_dimension(emb: EmbeddingFunction) -> int:
    dim = _resolve_sentence_transformer_dim(emb)
    if dim is not None:
        logging.info("Deduced embedding vector dimension automatically")
        return dim

    # Fallback to calling the embedding function
    logging.warning(
        "Cannot determine embedding vector dimension automatically, fallback to calling the embedding function..."
    )
    return len(emb(""))


def create_embedding_function_with_dimension(cfg: EmbeddingFunctionConfig):
    embedding_function = create_embedding_function(cfg)
    dim = _get_embedding_function_dimension(embedding_function)
    return (embedding_function, dim)


def _default_embedding_function(_: DefaultEmbeddingFunctionConfig) -> EmbeddingFunction:
    "Delegate to ONNXMiniLM_L6_V2"
    return create_embedding_function(SentenceTransformerEmbeddingFunctionConfig())


# Borrowed from chromadb/utils/embedding_functions/sentence_transformer_embedding_function.py,
# https://github.com/chroma-core/chroma/blob/019a25edac5eef393f139404f8f5b18fa85dd353/chromadb/utils/embedding_functions/sentence_transformer_embedding_function.py#L7
class SentenceTransformerEmbeddingFunction:
    models: ClassVar[Dict[str, "SentenceTransformer"]] = {}

    model_name: str
    device: str
    normalize_embeddings: bool
    kwargs: Dict[str, Any]

    _model: "SentenceTransformer"

    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ):
        """Initialize SentenceTransformerEmbeddingFunction.

        Args:
            model_name (str, optional): Identifier of the SentenceTransformer model, defaults to "all-MiniLM-L6-v2"
            device (str, optional): Device used for computation, defaults to "cpu"
            normalize_embeddings (bool, optional): Whether to normalize returned vectors, defaults to False
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
            ) from None

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        for key, value in kwargs.items():
            if not isinstance(value, (str, int, float, bool, list, dict, tuple)):
                raise ValueError(f"Keyword argument {key} is not a primitive type")
        self.kwargs = kwargs

        if model_name not in self.models:
            self.models[model_name] = SentenceTransformer(model_name, device=device, **kwargs)
        self._model = self.models[model_name]

    def __call__(self, input: Document) -> Embedding:
        """Generate embeddings for the given document.

        Args:
            input: Document to generate embeddings for.

        Returns:
            Embedding for the document.
        """
        # embeddings[input_idx]: embedding vector for sentences[input_idx]
        embeddings = self._model.encode(
            sentences=[input],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        embedding_vector: List[float] = embeddings.tolist()[0]
        return array("f", embedding_vector)

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> Self:
        model_name = config.get("model_name")
        device = config.get("device")
        normalize_embeddings = config.get("normalize_embeddings")
        kwargs = config.get("kwargs", {})

        assert not (model_name is None or device is None or normalize_embeddings is None), (
            "This code should not be reached"
        )

        return cls(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )


def _sentence_transformer_function(cfg: SentenceTransformerEmbeddingFunctionConfig) -> EmbeddingFunction:
    "Use sentence_transformers for embedding."
    return SentenceTransformerEmbeddingFunction(model_name=cfg.model_name)
