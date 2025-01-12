"""
Pinecone implementation of vector storage interface.
"""

import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec

from src.storage.base import (
    VectorStorageBase,
    VectorData,
    QueryResult,
    InitializationError,
    UpsertError,
    QueryError,
    DeleteError
)

logger = logging.getLogger(__name__)

class PineconeStorage(VectorStorageBase):
    """Pinecone implementation of vector storage."""

    def __init__(
        self,
        api_key: str,
        environment: str = "gcp-starter",
        cloud: str = "aws",
        region: str = "us-west-2",
        metric: str = "cosine"
    ):
        """
        Initialize Pinecone storage.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            cloud: Cloud provider for serverless
            region: Region for serverless
            metric: Distance metric to use
        """
        self.api_key = api_key
        self.environment = environment
        self.cloud = cloud
        self.region = region
        self.metric = metric
        self._client = None
        self._index = None

    @property
    def client(self) -> Pinecone:
        """Lazy initialization of Pinecone client."""
        if not self._client:
            self._client = Pinecone(api_key=self.api_key)
        return self._client

    async def initialize(self, dimension: int, index_name: str, **kwargs) -> None:
        """
        Initialize Pinecone index.

        Args:
            dimension: Vector dimension
            index_name: Name of the index to create/connect
            **kwargs: Additional initialization parameters

        Raises:
            InitializationError: If initialization fails
        """
        try:
            # Create ServerlessSpec
            spec = ServerlessSpec(
                cloud=self.cloud,
                region=self.region
            )

            # Check if index exists
            if index_name not in self.client.list_indexes().names():
                # Create index with specified configuration
                self.client.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=self.metric,
                    spec=spec
                )
                logger.info(f"Created new Pinecone index: {index_name}")
            else:
                logger.info(f"Connected to existing Pinecone index: {index_name}")

            # Connect to the index
            self._index = self.client.Index(index_name)

        except Exception as e:
            raise InitializationError(f"Failed to initialize Pinecone index: {str(e)}")

    async def upsert(
        self,
        vectors: List[VectorData],
        namespace: Optional[str] = None
    ) -> None:
        """
        Upsert vectors to Pinecone.

        Args:
            vectors: List of vectors to upsert
            namespace: Optional namespace for vectors

        Raises:
            UpsertError: If upsert operation fails
        """
        if not self._index:
            raise UpsertError("Index not initialized")

        try:
            # Convert VectorData objects to Pinecone format
            pinecone_vectors = [v.to_dict() for v in vectors]

            # Perform upsert operation
            self._index.upsert(
                vectors=pinecone_vectors,
                namespace=namespace
            )
            logger.debug(f"Upserted {len(vectors)} vectors to namespace: {namespace}")

        except Exception as e:
            raise UpsertError(f"Failed to upsert vectors: {str(e)}")

    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> QueryResult:
        """
        Query vectors in Pinecone.

        Args:
            vector: Query vector
            top_k: Number of results to return
            namespace: Optional namespace to query
            metadata_filter: Optional metadata filtering criteria
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results

        Returns:
            QueryResult containing matching vectors

        Raises:
            QueryError: If query operation fails
        """
        if not self._index:
            raise QueryError("Index not initialized")

        try:
            # Perform query operation
            response = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                filter=metadata_filter,
                include_metadata=include_metadata,
                include_values=include_values
            )

            # Convert response to VectorData objects
            vectors = []
            for match in response.matches:
                vectors.append(VectorData(
                    id=match.id,
                    values=match.values if include_values else [],
                    metadata=match.metadata if include_metadata else None,
                    score=match.score
                ))

            return QueryResult(vectors=vectors, namespace=namespace)

        except Exception as e:
            raise QueryError(f"Failed to query vectors: {str(e)}")

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorData]:
        """
        Fetch vectors by ID from Pinecone.

        Args:
            ids: List of vector IDs to fetch
            namespace: Optional namespace for vectors

        Returns:
            List of fetched vectors

        Raises:
            QueryError: If fetch operation fails
        """
        if not self._index:
            raise QueryError("Index not initialized")

        try:
            # Perform fetch operation
            response = self._index.fetch(ids=ids, namespace=namespace)

            # Convert response to VectorData objects
            vectors = []
            for vec_id, vec_data in response.vectors.items():
                vectors.append(VectorData(
                    id=vec_id,
                    values=vec_data.values,
                    metadata=vec_data.metadata
                ))

            return vectors

        except Exception as e:
            raise QueryError(f"Failed to fetch vectors: {str(e)}")

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> None:
        """
        Delete vectors from Pinecone.

        Args:
            ids: Optional list of vector IDs to delete
            namespace: Optional namespace for vectors
            delete_all: If True, delete all vectors in namespace

        Raises:
            DeleteError: If delete operation fails
        """
        if not self._index:
            raise DeleteError("Index not initialized")

        try:
            if delete_all:
                self._index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Deleted all vectors from namespace: {namespace}")
            elif ids:
                self._index.delete(ids=ids, namespace=namespace)
                logger.debug(f"Deleted {len(ids)} vectors from namespace: {namespace}")

        except Exception as e:
            raise DeleteError(f"Failed to delete vectors: {str(e)}")

    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces in Pinecone index.

        Returns:
            List of namespace names

        Raises:
            QueryError: If operation fails
        """
        if not self._index:
            raise QueryError("Index not initialized")

        try:
            stats = self._index.describe_index_stats()
            return list(stats.namespaces.keys())

        except Exception as e:
            raise QueryError(f"Failed to list namespaces: {str(e)}")

    async def get_stats(
        self,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics from Pinecone index.

        Args:
            namespace: Optional namespace to get stats for

        Returns:
            Dictionary of statistics

        Raises:
            QueryError: If operation fails
        """
        if not self._index:
            raise QueryError("Index not initialized")

        try:
            stats = self._index.describe_index_stats()
            if namespace:
                return stats.namespaces.get(namespace, {})
            return stats.to_dict()

        except Exception as e:
            raise QueryError(f"Failed to get statistics: {str(e)}")

    async def cleanup(self) -> None:
        """Cleanup Pinecone resources."""
        self._index = None
        self._client = None