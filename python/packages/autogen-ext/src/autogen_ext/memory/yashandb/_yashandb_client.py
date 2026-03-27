from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeAlias

import yaspy as yashandb
from typing_extensions import Self

from ._yashandb_configs import YashanDBVectorMemoryConfig
from ._yashandb_types import (
    ID,
    Document,
    EmbeddingFunction,
    Include,
    IncludeItem,
    Metadata,
    OneOrMany,
    YashanDBDistanceMetric,
    as_many,
    nonnull,
)

Connection: TypeAlias = yashandb.Connection
Cursor: TypeAlias = yashandb.Cursor


class YashanDBClient:
    _config: YashanDBVectorMemoryConfig
    _conn: Connection

    _embed: EmbeddingFunction
    _dim: int

    @property
    def _cursor(self) -> Cursor:
        return self._conn.cursor()

    def __init__(
        self,
        config: YashanDBVectorMemoryConfig,
        embed: EmbeddingFunction,
        dim: int,
    ):
        self._config = config
        self._conn = yashandb.connect(
            dsn=f"{config.host}:{config.port}",
            user=config.user,
            password=config.password,
        )
        self._conn.autocommit = True

        self._embed = embed
        self._dim = dim

    def get_collection(self, name: str) -> "YashanDBCollection":
        assert YashanDBCollection._check_collection_exists(self, name)

        return YashanDBCollection(
            name=name,
            vector_dimension=self._dim,
            embedding_function=self._embed,
            distance_metric=self._config.distance_metric,
            client=self,
            config=self._config,
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
            config=self._config,
        )

    def close(self) -> None:
        self._conn.close()


@dataclass
class YashanDBSingleQueryResult:
    ids: List[ID]
    documents: Optional[List[Document]]
    metadatas: Optional[List[Metadata]]
    distances: Optional[List[float]]


@dataclass
class YashanDBQueryResult:
    ids: List[List[ID]]
    documents: Optional[List[List[Document]]]
    metadatas: Optional[List[List[Metadata]]]
    distances: Optional[List[List[float]]]


@dataclass
class YashanDBGetResult:
    ids: List[ID]
    metadatas: Optional[List[Metadata]]
    documents: Optional[List[Document]]


# VECTOR_DISTANCE(lhs, rhs, DISTANCE_METRIC)
_VECTOR_DISTANCE_METRIC_MAPPING: Dict[YashanDBDistanceMetric, str] = {
    "cosine": "COSINE",
    "euclidean": "EUCLIDEAN",
    "l2_squared": "L2_SQUARED",
    "euclidean_squared": "EUCLIDEAN_SQUARED",
}

_PROJECTION_TYPE = OrderedDict[IncludeItem, Tuple[Optional[str], str]]


@dataclass
class YashanDBCollection:
    name: str
    vector_dimension: int
    embedding_function: EmbeddingFunction
    distance_metric: YashanDBDistanceMetric
    config: YashanDBVectorMemoryConfig

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
        config: YashanDBVectorMemoryConfig,
    ) -> Self:
        assert not cls._check_collection_exists(client, name)
        with client._cursor as c:
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS "{name}" (
                    id VARCHAR(36 CHAR) NOT NULL PRIMARY KEY, -- UUID
                    doc VARCHAR(65534) NOT NULL,
                    embedding VECTOR({vector_dimension}) NOT NULL,
                    meta JSON
                )""")

            index_name = f"idx_{name}_hnsw"

            # Check whether the index already exists
            c.execute(
                "SELECT COUNT(*) FROM user_indexes WHERE table_name = :table_name AND index_name = :index_name",
                {"table_name": name, "index_name": index_name},
            )
            index_exists: bool = c.fetchone()[0] == 1

            if not index_exists:
                c.execute(f"""
                    CREATE VECTOR INDEX "{index_name}" ON "{name}"(embedding)
                    ORGANIZATION NEIGHBOR GRAPH WITH DISTANCE COSINE
                    PARAMETERS(TYPE HNSW, M {config.m}, EFCONSTRUCTION {config.ef})
                """)

        return cls(
            name=name,
            vector_dimension=vector_dimension,
            embedding_function=embedding_function,
            distance_metric=distance_metric,
            client=client,
            config=config,
        )

    @classmethod
    def get_or_create_collection(
        cls,
        name: str,
        vector_dimension: int,
        embedding_function: EmbeddingFunction,
        distance_metric: YashanDBDistanceMetric,
        client: YashanDBClient,
        config: YashanDBVectorMemoryConfig,
    ) -> Self:
        create = cls if cls._check_collection_exists(client, name) else cls._create_collection

        return create(
            name=name,
            vector_dimension=vector_dimension,
            embedding_function=embedding_function,
            distance_metric=distance_metric,
            client=client,
            config=config,
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
                {"id": id, "doc": document, "embedding": embedding, "meta": metadata},
            )

    @classmethod
    def _get_projections(cls, include: Set[IncludeItem], projections: _PROJECTION_TYPE) -> str:
        """Generate the projection using include and projections.

        Projections is an ordered dict, whose key is the include item, and value is a tuple
            where [0] is the SQL expression, and [1] is the output column name.
            If SQL expression is not supplied, use the column name as the expression.

        Included columns are supplied as-is, using the projection's value as column name.

        Columns mentioned in projections but not included are also generated, but as a constant NULL value."""

        return ",".join(
            f"({(expr or column) if item in include else 'NULL'}) AS {column}"
            for item, (expr, column) in projections.items()
        )

    def _score_calculation(self, distance_expr: str) -> str:
        """Return the SQL expression that converts YashanDB distance to a similarity score."""
        # NOTE: This references chromadb's score calculation based on calculated distances.
        metric = self.config.distance_metric
        if metric == "cosine":
            return f"(1 - ({distance_expr}) / 2)"

        return f"(1 / (1 + ({distance_expr})))"

    def _query_single(
        self,
        query_text: Document,
        n_results: int = 10,
        score_threshold: Optional[float] = None,
        *,
        cursor: Cursor,
        include: Set[IncludeItem],
    ) -> YashanDBSingleQueryResult:
        metric = _VECTOR_DISTANCE_METRIC_MAPPING[self.distance_metric]
        distance_expr = f"VECTOR_DISTANCE(embedding, :query_embedding, {metric})"

        available_projections: _PROJECTION_TYPE = OrderedDict(
            (
                ("documents", (None, "doc")),
                ("metadatas", (None, "meta")),
                ("distances", (distance_expr, "distance")),
            )
        )

        query_embedding = self.embedding_function(query_text)
        sql = f"""
            SELECT id, {self._get_projections(include, available_projections)}
            FROM "{self.name}"
            WHERE ((:score_threshold IS NULL) OR (({self._score_calculation(distance_expr)}) >= :score_threshold))
            ORDER BY distance ASC
            FETCH APPROX FIRST :top_k ROWS ONLY
        """
        params = {"query_embedding": query_embedding, "top_k": n_results, "score_threshold": score_threshold}
        cursor.execute(sql, params)

        rows: List[Tuple[ID, Optional[Document], Optional[Metadata], Optional[float]]]
        rows = cursor.fetchall()

        return YashanDBSingleQueryResult(
            ids=[row[0] for row in rows],
            documents=[nonnull(row[1]) for row in rows] if "documents" in include else None,
            metadatas=[nonnull(row[2]) for row in rows] if "metadatas" in include else None,
            distances=[nonnull(row[3]) for row in rows] if "distances" in include else None,
        )

    def query(
        self,
        query_texts: OneOrMany[Document],
        n_results: int = 10,
        include: Optional[Include] = None,
        score_threshold: Optional[float] = None,
    ) -> YashanDBQueryResult:
        """
        Arguments:
        - query_texts - The document texts to get the closest neighbors of.
        - n_results - The number of neighbors to return for each query_texts. Default: 10.
        - include - A list of what to include in the results. Can contain "metadatas", "documents", "distances".
                    Ids are always included. Defaults to ["metadatas", "documents", "distances"].
        """

        if include is None:
            include = ["metadatas", "documents", "distances"]
        include_set = set(include)

        with self.client._cursor as c:
            results = [
                self._query_single(
                    query,
                    n_results=n_results,
                    cursor=c,
                    include=include_set,
                    score_threshold=score_threshold,
                )
                for query in as_many(query_texts)
            ]

            return YashanDBQueryResult(
                ids=[res.ids for res in results],
                documents=[nonnull(res.documents) for res in results] if "documents" in include_set else None,
                metadatas=[nonnull(res.metadatas) for res in results] if "metadatas" in include_set else None,
                distances=[nonnull(res.distances) for res in results] if "distances" in include_set else None,
            )

    def delete(self, ids: Optional[List[ID]] = None) -> None:
        # '(?,?,?,...,?)'
        if not ids:
            return

        placeholders = f"""({",".join(f"'{id}'" for id in ids)})"""
        with self.client._cursor as c:
            c.execute(f'DELETE FROM "{self.name}" WHERE id IN {placeholders}', ids)

    def get(self, include: Optional[Include] = None, limit: Optional[int] = None) -> YashanDBGetResult:
        """
        Arguments:
        - limit - The number of documents to return. Optional.
        - include - A list of what to include in the results. Can contain "metadatas", "documents".
                    Ids are always included. Defaults to ["metadatas", "documents"].
        """
        args = dict()

        if include is None:
            include = ["metadatas", "documents"]
        include_set = set(include)

        available_projections: _PROJECTION_TYPE = OrderedDict(
            (
                ("metadatas", (None, "meta")),
                ("documents", (None, "doc")),
            )
        )

        limit_clause: str = ""
        if limit is not None:
            limit_clause = "LIMIT :limit"
            args["limit"] = limit

        with self.client._cursor as c:
            c.execute(
                f"""
                    SELECT id, {self._get_projections(include_set, available_projections)} FROM "{self.name}"
                    {limit_clause}
                """,
                args,
            )
            rows: List[Tuple[ID, Optional[Metadata], Optional[Document]]] = c.fetchall()

            ids = [row[0] for row in rows]
            metadatas = [nonnull(row[1]) for row in rows] if "metadatas" in include_set else None
            documents = [nonnull(row[2]) for row in rows] if "documents" in include_set else None
            return YashanDBGetResult(ids=ids, metadatas=metadatas, documents=documents)

    def clear(self) -> None:
        with self.client._cursor as c:
            c.execute(f'TRUNCATE TABLE "{self.name}"')
