import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector store"""
    api_key: str
    environment: str
    tm_index_name: str = "tm-case-chunks"
    wipo_index_name: str = "wipo-mapping"

    @classmethod
    def from_env(cls) -> 'PineconeConfig':
        """Create configuration from environment variables"""
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")

        if not api_key or not environment:
            raise ValueError(
                "Missing required Pinecone environment variables: "
                "PINECONE_API_KEY and/or PINECONE_ENVIRONMENT"
            )

        return cls(
            api_key=api_key,
            environment=environment
        )

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    embedding_model_name: str = "all-mpnet-base-v2"  # Updated default
    embedding_dimension: int = 768

    def validate(self) -> None:
        """Validate model configuration"""
        if not self.embedding_model_name:
            raise ValueError("Embedding model name is required")
        if self.embedding_dimension <= 0:
            raise ValueError("Invalid embedding dimension")

@dataclass
class ProcessingConfig:
    """Configuration for text processing"""
    chunk_size: int = 1500  # Increased default chunk size
    chunk_overlap: int = 150
    max_retries: int = 3 # Added for Vertex AI retry logic (if needed in the future)
    wipo_confidence_threshold: float = 0.85 # Placeholder - You might want this for WIPO data handling

    def validate(self) -> None:
        """Validate processing configuration"""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI"""
    project_id: str
    location: str = "europe-west2"

    @classmethod
    def from_env(cls) -> 'VertexAIConfig':
        """Create configuration from environment variables"""
        project_id = os.getenv("VERTEX_PROJECT_ID")

        if not project_id:
            raise ValueError(
                "Missing required Vertex AI environment variable: VERTEX_PROJECT_ID"
            )

        return cls(
            project_id=project_id,
        )

class AppConfig:
    """Application configuration manager"""

    def __init__(
        self,
        pinecone_config: Optional[PineconeConfig] = None,
        model_config: Optional[ModelConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        vertex_ai_config: Optional[VertexAIConfig] = None
    ):
        """
        Initialize application configuration.

        Args:
            pinecone_config: Optional custom Pinecone configuration
            model_config: Optional custom model configuration
            processing_config: Optional custom processing configuration
            vertex_ai_config: Optional custom Vertex AI configuration
        """
        self.pinecone = pinecone_config or PineconeConfig.from_env()
        self.model = model_config or ModelConfig()
        self.processing = processing_config or ProcessingConfig()
        self.vertex_ai = vertex_ai_config or VertexAIConfig.from_env()

        # Data paths
        self.data_dir = Path("data")  # Assuming "data" directory in project root
        self.schema_path = self.data_dir / "schema.json"
        self.wipo_mapping_file = self.data_dir / "wipo_mapping.xlsx" # Placeholder, update if needed
        self.examples_path = self.data_dir / "examples.json"

        # Validate configuration
        self.validate()

    def validate(self) -> None:
        """Validate all configuration components"""
        self.model.validate()
        self.processing.validate()

        # Validate file paths
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        if not self.wipo_mapping_file.exists():
            raise FileNotFoundError(
                f"WIPO mapping file not found: {self.wipo_mapping_file}"
            )
        if not self.examples_path.exists():
            raise FileNotFoundError(
                f"Examples file not found: {self.examples_path}"
            )

    def load_schema(self) -> dict:
        """
        Load and validate JSON schema.

        Returns:
            Dict containing the schema

        Raises:
            FileNotFoundError: If schema file doesn't exist
            ValueError: If schema is invalid JSON
        """
        try:
            with open(self.schema_path) as f:
                schema = json.load(f)

            # Basic schema validation
            if not isinstance(schema, dict):
                raise ValueError("Invalid schema format")
            if "properties" not in schema:
                raise ValueError("Schema must define properties")

            return schema

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format"""
        return {
            "PINECONE_API_KEY": self.pinecone.api_key,
            "PINECONE_ENVIRONMENT": self.pinecone.environment,
            "PINECONE_INDEX_NAME": self.pinecone.tm_index_name,
            "WIPO_INDEX_NAME": self.pinecone.wipo_index_name,
            "EMBEDDING_MODEL_NAME": self.model.embedding_model_name,
            "EMBEDDING_DIMENSION": self.model.embedding_dimension,
            "CHUNK_SIZE": self.processing.chunk_size,
            "CHUNK_OVERLAP": self.processing.chunk_overlap,
            "MAX_RETRIES": self.processing.max_retries,  # Include max_retries
            "WIPO_CONFIDENCE_THRESHOLD": self.processing.wipo_confidence_threshold,
            "SCHEMA_PATH": str(self.schema_path),
            "WIPO_MAPPING_FILE": str(self.wipo_mapping_file),
            "EXAMPLES_PATH": str(self.examples_path),
            "VERTEX_PROJECT_ID": self.vertex_ai.project_id,
            "VERTEX_LOCATION": self.vertex_ai.location,
        }

# Global configuration instance
_config: Optional[AppConfig] = None

def get_config() -> Dict[str, Any]:
    """
    Get application configuration.

    Returns:
        Dictionary containing configuration values

    Raises:
        ConfigurationError: If configuration initialization fails
    """
    global _config

    if _config is None:
        try:
            _config = AppConfig()
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {str(e)}")

    return _config.to_dict()

def load_schema(schema_path: str) -> dict:
    """
    Load schema from specified path.

    Args:
        schema_path: Path to schema file

    Returns:
        Dictionary containing the schema

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema is invalid
    """
    global _config

    if _config is None:
        _config = AppConfig()

    return _config.load_schema()