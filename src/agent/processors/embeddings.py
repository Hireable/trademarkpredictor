"""
Embeddings processor for generating and managing text embeddings.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

from src.exceptions import ProcessingError
from src.agent.processors.document import DocumentChunk
from src.config.base import ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingVector:
    """Represents an embedding vector with its metadata."""
    vector: List[float]
    text: str
    metadata: Optional[Dict[str, Any]] = None
    vector_id: Optional[str] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
        if self.vector_id is None:
            # Create a unique ID based on text content
            self.vector_id = f"vec-{hash(self.text)}"

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        return len(self.vector)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage."""
        return {
            "id": self.vector_id,
            "values": self.vector,
            "metadata": {
                "text": self.text,
                **self.metadata
            }
        }

class EmbeddingsProcessor:
    """Handles generation and processing of text embeddings."""

    def __init__(self, config: ModelConfig):
        """
        Initialize the embeddings processor.

        Args:
            config: Model configuration containing embedding settings
        """
        self.config = config
        self._model = None
        self._batch_size = 32  # Default batch size for embedding generation

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of the embedding model."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.config.embedding_model_name)
            except Exception as e:
                raise ProcessingError(f"Failed to load embedding model: {str(e)}")
        return self._model

    async def generate_embeddings(
        self,
        texts: Union[str, List[str], DocumentChunk, List[DocumentChunk]]
    ) -> Union[EmbeddingVector, List[EmbeddingVector]]:
        """
        Generate embeddings for text or document chunks.

        Args:
            texts: Input text(s) or document chunk(s)

        Returns:
            Single EmbeddingVector or list of EmbeddingVectors

        Raises:
            ProcessingError: If embedding generation fails
        """
        try:
            # Convert input to list of texts and preserve metadata
            if isinstance(texts, str):
                text_list = [texts]
                metadata_list = [None]
            elif isinstance(texts, DocumentChunk):
                text_list = [texts.text]
                metadata_list = [texts.metadata]
            elif isinstance(texts, list):
                if all(isinstance(t, str) for t in texts):
                    text_list = texts
                    metadata_list = [None] * len(texts)
                elif all(isinstance(t, DocumentChunk) for t in texts):
                    text_list = [t.text for t in texts]
                    metadata_list = [t.metadata for t in texts]
                else:
                    raise ValueError("List must contain either all strings or all DocumentChunks")
            else:
                raise ValueError("Unsupported input type")

            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(text_list), self._batch_size):
                batch_texts = text_list[i:i + self._batch_size]
                batch_metadata = metadata_list[i:i + self._batch_size]
                
                # Generate embeddings for batch
                batch_vectors = self.model.encode(
                    batch_texts,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )

                # Create EmbeddingVector objects
                batch_embeddings = [
                    EmbeddingVector(
                        vector=vector.tolist(),
                        text=text,
                        metadata=metadata
                    )
                    for vector, text, metadata in zip(batch_vectors, batch_texts, batch_metadata)
                ]
                all_embeddings.extend(batch_embeddings)

            # Return single vector or list based on input type
            if isinstance(texts, (str, DocumentChunk)):
                return all_embeddings[0]
            return all_embeddings

        except Exception as e:
            raise ProcessingError(f"Failed to generate embeddings: {str(e)}")

    def compute_similarity(
        self,
        vector1: Union[List[float], EmbeddingVector],
        vector2: Union[List[float], EmbeddingVector]
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vector1: First vector or EmbeddingVector
            vector2: Second vector or EmbeddingVector

        Returns:
            Cosine similarity score
        """
        # Extract raw vectors if EmbeddingVector objects provided
        if isinstance(vector1, EmbeddingVector):
            vector1 = vector1.vector
        if isinstance(vector2, EmbeddingVector):
            vector2 = vector2.vector

        # Convert to numpy arrays
        v1 = np.array(vector1)
        v2 = np.array(vector2)

        # Compute cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def average_embeddings(
        self,
        vectors: List[Union[List[float], EmbeddingVector]]
    ) -> List[float]:
        """
        Compute average of multiple embedding vectors.

        Args:
            vectors: List of vectors or EmbeddingVector objects

        Returns:
            Averaged embedding vector
        """
        if not vectors:
            raise ValueError("Cannot compute average of empty vector list")

        # Extract raw vectors if EmbeddingVector objects provided
        raw_vectors = []
        for vec in vectors:
            if isinstance(vec, EmbeddingVector):
                raw_vectors.append(vec.vector)
            else:
                raw_vectors.append(vec)

        # Convert to numpy array and compute mean
        array_vectors = np.array(raw_vectors)
        return np.mean(array_vectors, axis=0).tolist()

    def normalize_vector(
        self,
        vector: Union[List[float], EmbeddingVector]
    ) -> List[float]:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector or EmbeddingVector

        Returns:
            Normalized vector
        """
        if isinstance(vector, EmbeddingVector):
            vector = vector.vector

        vec_array = np.array(vector)
        norm = np.linalg.norm(vec_array)
        
        if norm == 0:
            return [0.0] * len(vector)
            
        return (vec_array / norm).tolist()

    async def cleanup(self):
        """Cleanup resources."""
        self._model = None