from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, TypeAlias, TypeVar, Union, cast

if TYPE_CHECKING:
    from array import array


T = TypeVar("T")
OneOrMany = Union[T, List[T]]


def as_many(values: OneOrMany[T]) -> List[T]:
    return values if isinstance(values, list) else [values]


YashanDBDistanceMetric = Literal["cosine", "euclidean", "euclidean_squared", "l2_squared"]

ID = str
Metadata = Dict[str, Any]
Document = str
Embedding: TypeAlias = "array[float]"

EmbeddingFunction = Callable[[Document], Embedding]

IncludeItem = Literal["documents", "embeddings", "metadatas", "distances"]
Include = List[IncludeItem]


def nonnull(x: Optional[T]) -> T:
    return cast(T, x)
