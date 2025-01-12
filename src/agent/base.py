"""
Base agent class providing core functionality for trademark processing.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import vertexai
from sentence_transformers import SentenceTransformer
from vertexai.generative_models import GenerativeModel

from ..exceptions import ConfigurationError, ResourceError
from ..config.base import ModelConfig, ProcessingConfig, VertexAIConfig
from ..config.settings import get_settings
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class ResourceManager:
    """Manages initialization and cleanup of heavy resources."""
    
    config: Dict[str, Any]
    _embedding_model = None
    _gemini_model = None
    
    @property
    def embedding_model(self):
        """Lazy loading of embedding model."""
        if not self._embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    self.config["EMBEDDING_MODEL_NAME"]
                )
                logger.info(f"Initialized embedding model: {self.config['EMBEDDING_MODEL_NAME']}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {str(e)}")
                raise ResourceError(f"Failed to initialize embedding model: {str(e)}")
        return self._embedding_model
    
    @property
    def gemini_model(self):
        """Lazy loading of Gemini model."""
        if not self._gemini_model:
            try:
                vertexai.init(
                    project=self.config["VERTEX_PROJECT_ID"],
                    location=self.config["VERTEX_LOCATION"]
                )
                self._gemini_model = GenerativeModel("gemini-1.5-pro-002")
                logger.info("Initialized Gemini model")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {str(e)}")
                raise ResourceError(f"Failed to initialize Gemini model: {str(e)}")
        return self._gemini_model
    
    async def cleanup(self):
        """Cleanup resources."""
        # Add any necessary cleanup logic here
        self._embedding_model = None
        self._gemini_model = None
        logger.info("Cleaned up resources")


class BaseAgent(ABC):
    """
    Abstract base class for document processing agents.
    
    This class provides the core functionality and interface that specific
    implementations (like TrademarkAgent) must follow.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        processing_config: Optional[ProcessingConfig] = None,
        vertex_ai_config: Optional[VertexAIConfig] = None,
        model_config: Optional[ModelConfig] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            config: Optional custom configuration dictionary
            processing_config: Optional custom processing configuration
            vertex_ai_config: Optional custom Vertex AI configuration
            model_config: Optional custom model configuration
            
        Raises:
            ConfigurationError: If required configuration is missing or invalid
        """
        # Get settings and merge with any custom config
        settings = get_settings()
        self.config = config or {
            "PINECONE_INDEX_NAME": settings.pinecone.chunks_index,
            "EMBEDDING_MODEL_NAME": settings.vertex_ai.model_name,
            "VERTEX_PROJECT_ID": settings.vertex_ai.project_id,
            "VERTEX_LOCATION": settings.vertex_ai.location
        }
        
        self._validate_config()
        
        self.processing_config = processing_config or ProcessingConfig()
        self.vertex_ai_config = vertex_ai_config or VertexAIConfig.from_env()
        self.model_config = model_config or ModelConfig()
        
        # Initialize resource manager
        self.resources = ResourceManager(self.config)
        
        # Initialize session to None - will be created when needed
        self.session = None
        
        logger.info(f"{self.__class__.__name__} initialized with configuration")
    
    def _validate_config(self) -> None:
        """Validate the configuration."""
        required_keys = [
            "PINECONE_INDEX_NAME",
            "EMBEDDING_MODEL_NAME",
            "VERTEX_PROJECT_ID",
            "VERTEX_LOCATION"
        ]
        
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            error_msg = f"Missing required configuration keys: {missing_keys}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        logger.debug("Configuration validated successfully")
    
    @abstractmethod
    async def process_document(self, document_path: str, **kwargs) -> Any:
        """
        Process a document. Must be implemented by concrete classes.
        
        Args:
            document_path: Path to the document to process
            **kwargs: Additional arguments specific to the implementation
            
        Returns:
            Implementation-specific processing results
        """
        pass
    
    @abstractmethod
    async def generate_embeddings(self, text: str) -> list:
        """
        Generate embeddings for the given text.
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List of embeddings
        """
        pass
    
    @abstractmethod
    async def store_embeddings(self, embeddings: list, metadata: Dict[str, Any]) -> None:
        """
        Store embeddings in the vector store.
        
        Args:
            embeddings: List of embeddings to store
            metadata: Metadata to store with the embeddings
        """
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources when done."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            await self.resources.cleanup()
            logger.info(f"{self.__class__.__name__} cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise ResourceError(f"Failed to clean up resources: {str(e)}")

    async def __aenter__(self):
        """Support for async context manager protocol."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support for async context manager protocol."""
        await self.cleanup()