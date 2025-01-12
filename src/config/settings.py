from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    FilePath,
    DirectoryPath,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PineconeSettings(BaseModel):
    """
    Settings specific to Pinecone vector database configuration.
    We separate these settings into their own model for better organization
    and to enable specific validation rules for Pinecone-related settings.
    """
    api_key: str = Field(..., min_length=32, description="Pinecone API key for authentication")
    environment: str = Field(..., description="Pinecone environment (e.g., gcp-starter)")
    chunks_index: str = Field("tm-case-chunks", description="Index name for storing case chunks")
    predictive_index: str = Field("tm-predictor-index", description="Index for predictive data")
    chunks_dimension: int = Field(1024, description="Vector dimension for case chunks")
    predictive_dimension: int = Field(1024, description="Vector dimension for predictive data")
    batch_size: int = Field(100, ge=1, description="Batch size for Pinecone operations")
    max_connections: int = Field(10, ge=1, description="Maximum concurrent connections")
    timeout_seconds: int = Field(30, ge=1, description="Operation timeout in seconds")
    location: str = Field("europe-west4", description="Pinecone server location")

    model_config = ConfigDict(
        case_sensitive=True,
        validate_assignment=True
    )

class VertexAISettings(BaseModel):
    """
    Settings for Google Vertex AI configuration.
    Encapsulates all AI model-related settings and their validation rules.
    """
    project_id: str = Field(..., description="Google Cloud project ID")
    location: str = Field("europe-west2", description="Vertex AI location")
    model_name: str = Field("gemini-1.5-pro", description="Model identifier")
    max_tokens: int = Field(4000, ge=1, description="Maximum output tokens")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="Generation temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")

    model_config = ConfigDict(
        case_sensitive=True,
        validate_assignment=True
    )

class ProcessingSettings(BaseModel):
    """
    Settings for document processing configuration.
    Controls how documents are chunked and processed.
    """
    chunk_size: int = Field(1500, ge=100, description="Size of text chunks")
    chunk_overlap: int = Field(150, ge=0, description="Overlap between chunks")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    max_workers: int = Field(8, ge=1, description="Maximum worker processes")
    worker_threads: int = Field(2, ge=1, description="Worker threads per process")
    memory_limit_mb: int = Field(8192, ge=1024, description="Memory limit in MB")
    chunk_memory_limit_mb: int = Field(512, ge=64, description="Per-chunk memory limit")

    @field_validator("chunk_overlap")
    def validate_chunk_overlap(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure chunk overlap is smaller than chunk size."""
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

class Settings(BaseSettings):
    """
    Main settings class that brings together all configuration components.
    Uses Pydantic's BaseSettings for automatic environment variable loading
    and validation.
    """
    # Base paths configuration
    data_dir: DirectoryPath = Field(Path("data"), description="Base data directory")
    pdf_dir: DirectoryPath = Field(Path("data/raw_pdfs"), description="PDF storage directory")
    output_dir: DirectoryPath = Field(Path("data/processed"), description="Output directory")
    log_dir: DirectoryPath = Field(Path("data/logs"), description="Log files directory")
    
    # Schema and data files
    schema_path: FilePath = Field(..., description="Path to JSON schema file")
    examples_path: FilePath = Field(..., description="Path to examples file")
    
    # Component-specific settings
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    vertex_ai: VertexAISettings = Field(default_factory=VertexAISettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    
    # Logging configuration
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("detailed", description="Log format (basic, detailed, or json)")
    max_log_size_mb: int = Field(100, ge=1, description="Maximum log file size")
    log_backup_count: int = Field(5, ge=0, description="Number of log backups to keep")

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True
    )

    def create_directories(self) -> None:
        """
        Create necessary directories if they don't exist.
        This ensures all required paths are available before processing starts.
        """
        for directory in [self.data_dir, self.pdf_dir, self.output_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @field_validator("schema_path", "examples_path")
    def validate_file_exists(cls, v: Path) -> Path:
        """Ensure required files exist."""
        if not v.exists():
            raise ValueError(f"File not found: {v}")
        return v

@lru_cache
def get_settings() -> Settings:
    """
    Get settings instance with caching.
    Using lru_cache ensures we only load and validate settings once,
    improving performance in applications that frequently access settings.
    
    Returns:
        Settings instance with validated configuration
    """
    settings = Settings()
    settings.create_directories()
    return settings