"""
Base configuration classes for the trademark predictor application.
All core configuration models and validation logic.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator
import os

class BaseConfig(BaseModel):
    """Base configuration class with common settings."""
    
    debug: bool = Field(False, description="Enable debug mode")
    env: str = Field("production", description="Environment (development/production)")

    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"  # Prevent extra attributes

class StorageConfig(BaseConfig):
    """Configuration for vector storage."""
    
    api_key: str = Field(..., description="Storage service API key")
    environment: str = Field("gcp-starter", description="Storage environment")
    index_name: str = Field(..., description="Vector index name")
    dimension: int = Field(1024, description="Vector dimension")
    metric: str = Field("cosine", description="Distance metric")
    batch_size: int = Field(100, description="Batch operation size")

class ModelConfig(BaseConfig):
    """Configuration for ML models."""
    
    embedding_model_name: str = Field(
        "all-mpnet-base-v2",
        description="Name of embedding model"
    )
    embedding_dimension: int = Field(1024, description="Embedding vector dimension")
    max_sequence_length: int = Field(512, description="Maximum sequence length")
    cache_dir: Optional[Path] = Field(None, description="Model cache directory")

class ProcessingConfig(BaseConfig):
    """Configuration for document processing."""
    
    chunk_size: int = Field(1500, ge=100, description="Size of text chunks")
    chunk_overlap: int = Field(150, ge=0, description="Overlap between chunks")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    batch_size: int = Field(5, ge=1, description="Processing batch size")
    embedding_dimension: int = Field(1024, ge=1, description="Embedding dimension")

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure chunk overlap is smaller than chunk size."""
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

class VertexAIConfig(BaseConfig):
    """Configuration for Vertex AI."""
    
    project_id: str = Field(..., description="Google Cloud project ID")
    location: str = Field("europe-west2", description="Vertex AI location")
    model_name: str = Field("gemini-1.5-pro", description="Model name")
    max_tokens: int = Field(4000, ge=1, description="Maximum output tokens")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="Generation temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")
    timeout: int = Field(300, ge=1, description="API timeout in seconds")
    
    @classmethod
    def from_env(cls) -> "VertexAIConfig":
        """Create config from environment variables."""
        return cls(
            project_id=os.getenv("VERTEX_PROJECT_ID", ""),
            location=os.getenv("VERTEX_LOCATION", "europe-west2"),
            model_name=os.getenv("VERTEX_MODEL_NAME", "gemini-1.5-pro"),
            max_tokens=int(os.getenv("VERTEX_MAX_TOKENS", "4000")),
            temperature=float(os.getenv("VERTEX_TEMPERATURE", "0.2")),
            top_p=float(os.getenv("VERTEX_TOP_P", "0.95")),
            timeout=int(os.getenv("VERTEX_TIMEOUT", "300"))
        )

class LoggingConfig(BaseConfig):
    """Configuration for logging."""
    
    level: str = Field("INFO", description="Logging level")
    format: str = Field("detailed", description="Log format style")
    file: Optional[Path] = Field(None, description="Log file path")
    max_size: int = Field(10 * 1024 * 1024, description="Max log file size in bytes")
    backup_count: int = Field(5, description="Number of backup logs to keep")

def load_config(config_path: Optional[Path] = None) -> Dict[str, BaseConfig]:
    """
    Load configuration from file and/or environment variables.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Dictionary containing configuration objects
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Load from file if provided
    file_config = {}
    if config_path and config_path.exists():
        import json
        with open(config_path) as f:
            file_config = json.load(f)
    
    # Create configuration objects
    return {
        "storage": StorageConfig(**file_config.get("storage", {})),
        "model": ModelConfig(**file_config.get("model", {})),
        "processing": ProcessingConfig(**file_config.get("processing", {})),
        "vertex": VertexAIConfig.from_env(),
        "logging": LoggingConfig(**file_config.get("logging", {}))
    }