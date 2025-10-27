from array import array
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yasdb
import yasdb.cursor
from typing_extensions import Self

from ._yashandb_configs import YashanDBVectorMemoryConfig
from ._yashandb_types import ID, Document, EmbeddingFunction, Metadata, OneOrMany, YashanDBDistanceMetric, as_many


class YashanDBClient:
    _config: YashanDBVectorMemoryConfig
    _conn: yasdb.YasdbConnection

    @property
    def _cursor(self) -> yasdb.cursor.YasdbCursor:
        return self._conn.cursor()

    def __init__(self, config: YashanDBVectorMemoryConfig):
        self._config = config
        self._conn = yasdb.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            autocommit=True,
        )

    def get_or_create_collection(
        self,
        name: str,
        vector_dimension: int,
        embedding_function: EmbeddingFunction,
        distance_metric: YashanDBDistanceMetric,
    ) -> "YashanDBCollection":
        return YashanDBCollection.get_or_create_collection(
            name=name,
            vector_dimension=vector_dimension,
            embedding_function=embedding_function,
            distance_metric=distance_metric,
            client=self,
        )


@dataclass
class YashanDBSingleQueryResult:
    ids: List[ID]
    documents: List[Document]
    metadatas: List[Metadata]
    distances: List[float]


@dataclass
class YashanDBQueryResult:
    ids: List[List[ID]]
    documents: List[List[Document]]
    metadatas: List[List[Metadata]]
    distances: List[List[float]]


@dataclass
class YashanDBGetResult:
    ids: List[ID]


# VECTOR_DISTANCE(lhs, rhs, DISTANCE_METRIC)
_VECTOR_DISTANCE_METRIC_MAPPING: Dict[YashanDBDistanceMetric, str] = {
    "cosine": "COSINE",
    "euclidean": "EUCLIDEAN",
    "l2_squared": "L2_SQUARED",
    "euclidean_squared": "EUCLIDEAN_SQUARED",
}


@dataclass
class YashanDBCollection:
    name: str
    vector_dimension: int
    embedding_function: EmbeddingFunction
    distance_metric: YashanDBDistanceMetric

    client: YashanDBClient

    @classmethod
    def _check_collection_exists(cls, client: YashanDBClient, name: str) -> bool:
        # sanity check on table names
        assert name.isascii() and name.isprintable() and not (set(name) & {'"', "'", "\\"}), (
            f'Collection name should not contain exotic characters, got "{name}"'
        )

        with client._cursor as c:
            try:
                c.execute("SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME = :table_name", {"table_name": name})
                row: Tuple[int] = c.fetchone()
                return row[0] == 1
            except Exception as e:
                raise RuntimeError("Failed to check collection") from e

    @classmethod
    def _create_collection(
        cls,
        name: str,
        vector_dimension: int,
        embedding_function: EmbeddingFunction,
        distance_metric: YashanDBDistanceMetric,
        client: YashanDBClient,
    ) -> Self:
        assert not cls._check_collection_exists(client, name)
        with client._cursor as c:
            c.execute(f"""
                CREATE TABLE "{name}" (
                    id VARCHAR(36 CHAR) NOT NULL PRIMARY KEY, -- UUID
                    doc VARCHAR(8192 BYTE) NOT NULL,
                    embedding VECTOR({vector_dimension}) NOT NULL,
                    meta JSON NOT NULL
                )""")

        return cls(
            name=name,
            vector_dimension=vector_dimension,
            embedding_function=embedding_function,
            distance_metric=distance_metric,
            client=client,
        )

    @classmethod
    def get_or_create_collection(
        cls,
        name: str,
        vector_dimension: int,
        embedding_function: EmbeddingFunction,
        distance_metric: YashanDBDistanceMetric,
        client: YashanDBClient,
    ) -> Self:
        create = cls if cls._check_collection_exists(client, name) else cls._create_collection

        return create(
            name=name,
            vector_dimension=vector_dimension,
            embedding_function=embedding_function,
            distance_metric=distance_metric,
            client=client,
        )

    def add(
        self,
        id: ID,
        document: Document,
        metadata: Optional[Metadata] = None,
    ) -> None:
        embedding = self.embedding_function(document)

        with self.client._cursor as c:
            c.execute(
                f'INSERT INTO "{self.name}"(id, doc, embedding, meta) VALUES (:id, :doc, :embedding, :meta)',
                {"id": id, "doc": document, "embedding": embedding, "meta": metadata or {}},
            )

    def _query_single(
        self,
        query_text: Document,
        n_results: int = 10,
        # include: Include = ["metadatas", "documents", "distances"],
        *,
        cursor: yasdb.cursor.YasdbCursor,
    ) -> YashanDBSingleQueryResult:
        metric = _VECTOR_DISTANCE_METRIC_MAPPING[self.distance_metric]
        query_embedding = self.embedding_function(query_text)
        cursor.execute(
            f"""SELECT id, doc, meta
                        , VECTOR_DISTANCE(embedding, :query_embedding, {metric}) distance
                    FROM "{self.name}"
                    ORDER BY distance ASC
                    FETCH APPROX FIRST :top_k ROWS ONLY""",
            {"query_embedding": query_embedding, "top_k": n_results},
        )

        rows: List[Tuple[ID, Document, Metadata, float]] = cursor.fetchall()
        return YashanDBSingleQueryResult(
            ids=[row[0] for row in rows],
            documents=[row[1] for row in rows],
            metadatas=[row[2] for row in rows],
            distances=[row[3] for row in rows],
        )

    def query(
        self,
        query_texts: OneOrMany[Document],
        n_results: int = 10,
        # include: Include = ["metadatas", "documents", "distances"],
    ) -> YashanDBQueryResult:
        with self.client._cursor as c:
            results = [self._query_single(query, n_results=n_results, cursor=c) for query in as_many(query_texts)]

            return YashanDBQueryResult(
                ids=[res.ids for res in results],
                documents=[res.documents for res in results],
                metadatas=[res.metadatas for res in results],
                distances=[res.distances for res in results],
            )

    def delete(self, ids: Optional[List[ID]] = None) -> None:
        # '(?,?,?,...,?)'
        if not ids:
            return

        placeholders = f"""({",".join(f"'{id}'" for id in ids)})"""
        with self.client._cursor as c:
            c.execute(f'DELETE FROM "{self.name}" WHERE id IN {placeholders}', ids)

    def get(self) -> YashanDBGetResult:
        with self.client._cursor as c:
            c.execute(f'SELECT id FROM "{self.name}"')
            rows: List[Tuple[ID]] = c.fetchall()

            ids = [row[0] for row in rows]
            return YashanDBGetResult(ids=ids)
