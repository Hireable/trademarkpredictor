"""
Base interface for vector storage implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class VectorData:
    """Container for vector data and metadata."""
    id: str
    values: List[float]
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None  # For query results

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "id": self.id,
            "values": self.values
        }
        if self.metadata:
            result["metadata"] = self.metadata
        if self.score is not None:
            result["score"] = self.score
        return result

@dataclass
class QueryResult:
    """Container for query results."""
    vectors: List[VectorData]
    namespace: Optional[str] = None
    
    @property
    def ids(self) -> List[str]:
        """Get list of vector IDs."""
        return [v.id for v in self.vectors]
    
    @property
    def scores(self) -> List[float]:
        """Get list of similarity scores."""
        return [v.score for v in self.vectors if v.score is not None]

class VectorStorageBase(ABC):
    """Abstract base class for vector storage implementations."""

    @abstractmethod
    async def initialize(self, dimension: int, **kwargs) -> None:
        """
        Initialize the vector storage.

        Args:
            dimension: Dimension of vectors to store
            **kwargs: Implementation-specific initialization parameters
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        vectors: List[VectorData],
        namespace: Optional[str] = None
    ) -> None:
        """
        Insert or update vectors in storage.

        Args:
            vectors: List of vectors to upsert
            namespace: Optional namespace for vectors
        """
        pass

    @abstractmethod
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
        Query vectors by similarity.

        Args:
            vector: Query vector
            top_k: Number of results to return
            namespace: Optional namespace to query
            metadata_filter: Optional metadata filtering criteria
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results

        Returns:
            QueryResult containing matching vectors
        """
        pass

    @abstractmethod
    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorData]:
        """
        Fetch vectors by ID.

        Args:
            ids: List of vector IDs to fetch
            namespace: Optional namespace for vectors

        Returns:
            List of fetched vectors
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> None:
        """
        Delete vectors from storage.

        Args:
            ids: Optional list of vector IDs to delete
            namespace: Optional namespace for vectors
            delete_all: If True, delete all vectors in namespace
        """
        pass

    @abstractmethod
    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces in storage.

        Returns:
            List of namespace names
        """
        pass

    @abstractmethod
    async def get_stats(
        self,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get storage statistics.

        Args:
            namespace: Optional namespace to get stats for

        Returns:
            Dictionary of statistics
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

class VectorStorageError(Exception):
    """Base exception class for vector storage errors."""
    pass

class InitializationError(VectorStorageError):
    """Raised when storage initialization fails."""
    pass

class UpsertError(VectorStorageError):
    """Raised when upserting vectors fails."""
    pass

class QueryError(VectorStorageError):
    """Raised when querying vectors fails."""
    pass

class DeleteError(VectorStorageError):
    """Raised when deleting vectors fails."""
    pass