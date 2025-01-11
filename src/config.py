import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging
from functools import lru_cache

from dotenv import load_dotenv
from jsonschema import validate, ValidationError as JsonSchemaValidationError

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SchemaConfig:
    """Enhanced configuration for schema validation and metadata enrichment"""
    schema_path: Path
    enable_validation: bool = True
    strict_mode: bool = False
    allow_additional_properties: bool = False
    validation_cache_size: int = 1000
    
    # Confidence thresholds for predictive fields
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "mark_similarity": 0.7,
        "gds_sim": 0.75,
        "outcome": 0.8
    })
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        # Load and validate schema structure
        self._schema_data = self._load_schema()
        self._validate_schema_structure()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema from file without caching"""
        try:
            with open(self.schema_path) as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load schema: {str(e)}")
    
    @property
    def schema(self) -> Dict[str, Any]:
        """Access the loaded schema data"""
        return self._schema_data
    
    def _validate_schema_structure(self):
        """Validate the schema has required components"""
        required_keys = ["type", "properties", "required"]
        missing = [key for key in required_keys if key not in self.schema]
        if missing:
            raise ValueError(f"Schema missing required keys: {missing}")
            
        if self.schema["type"] != "object":
            raise ValueError("Schema root must be of type 'object'")

@dataclass
class MetadataConfig:
    """Configuration for metadata enrichment and processing"""
    namespace_strategy: str = "jurisdiction"  # How to partition data
    batch_size: int = 100
    embedding_dimension: int = 1024
    
    # Predictive field configuration
    predictive_fields: List[str] = field(default_factory=lambda: [
        "mark_similarity",
        "gds_sim",
        "outcome"
    ])
    
    # Feature extraction settings
    feature_config: Dict[str, Any] = field(default_factory=lambda: {
        "use_gpu": False,
        "max_sequence_length": 512,
        "pooling_strategy": "mean"
    })
    
    def validate(self):
        """Validate metadata configuration"""
        valid_strategies = ["jurisdiction", "date", "custom"]
        if self.namespace_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid namespace strategy. Must be one of: {valid_strategies}"
            )
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
            
        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

@dataclass
class PineconeConfig:
    """Enhanced Pinecone configuration with metadata support"""
    api_key: str = "pcsk_66P79y_NY84eEkPV9rGvCr9G34d4pghJ95bzBaS3ay7fi7zYkghND3Xnj5qfoLWFdBKhwU"
    environment: str = "gcp-starter"
    document_index_name: str = "tm-doc-chunks"
    metadata_index_name: str = "tm-case-metadata"
    cloud: str = "gcp-starter"
    region: str = "europe-west4"
    
    # Metadata indexing configuration
    metadata_fields: List[str] = field(default_factory=lambda: [
        "jurisdiction",
        "case_ref",
        "dec_date"
    ])
    
    def validate(self):
        """Validate Pinecone configuration"""
        if not self.api_key or len(self.api_key) < 32:
            raise ValueError("Invalid Pinecone API key")
        if not self.environment:
            raise ValueError("Pinecone environment must be specified")
        if not self.document_index_name or not self.metadata_index_name:
            raise ValueError("Both document and metadata index names required")

@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI services"""
    project_id: str
    location: str = "europe-west2"
    model_name: str = "gemini-1.5-pro"  # Updated to match environment variable
    temperature: float = 0.2
    max_output_tokens: int = 1024
    top_p: float = 0.8
    top_k: int = 40

    @classmethod
    def from_env(cls) -> 'VertexAIConfig':
        """Create configuration from environment variables"""
        return cls(
            project_id=os.getenv('VERTEX_PROJECT_ID'),
            location=os.getenv('VERTEX_LOCATION', 'europe-west2'),
            model_name=os.getenv('VERTEX_MODEL_NAME', 'gemini-1.5-pro')
        )

    def validate(self):
        """Validate the configuration"""
        if not self.project_id:
            raise ValueError("Vertex AI project ID must be specified")
        if not self.location:
            raise ValueError("Vertex AI location must be specified")
        if not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        if not 0 <= self.top_p <= 1:
            raise ValueError("Top P must be between 0 and 1")

@dataclass
class ProcessingConfig:
    """Enhanced processing configuration"""
    chunk_size: int = 1500
    chunk_overlap: int = 150
    max_retries: int = 3
    batch_size: int = 5
    
    # New validation settings
    validation_workers: int = 2
    max_validation_queue: int = 1000
    skip_invalid_chunks: bool = True
    
    def validate(self):
        """Validate processing configuration"""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if self.max_retries <= 0:
            raise ValueError("Max retries must be positive")
        if self.validation_workers <= 0:
            raise ValueError("Must have at least one validation worker")

class AppConfig:
    """Enhanced application configuration manager"""
    
    def __init__(
        self,
        schema_config: Optional[SchemaConfig] = None,
        metadata_config: Optional[MetadataConfig] = None,
        pinecone_config: Optional[PineconeConfig] = None,
        processing_config: Optional[ProcessingConfig] = None
    ):
        """Initialize with comprehensive configuration components"""
        # Load environment variables
        load_dotenv()
        
        # Initialize configurations
        self.schema = schema_config or self._init_schema_config()
        self.metadata = metadata_config or MetadataConfig()
        self.pinecone = pinecone_config or self._init_pinecone_config()
        self.processing = processing_config or ProcessingConfig()
        
        # Validate all components
        self.validate()
        
        # Set up data paths
        self.data_dir = Path(os.getenv("DATA_DIR", "data"))
        self.examples_path = Path(os.getenv("EXAMPLES_PATH", 
                                          str(self.data_dir / "examples.json")))
    
    def _init_schema_config(self) -> SchemaConfig:
        """Initialize schema configuration from environment"""
        schema_path = Path(os.getenv("SCHEMA_PATH", "schema.json"))
        enable_validation = os.getenv("ENABLE_SCHEMA_VALIDATION", "true").lower() == "true"
        strict_mode = os.getenv("STRICT_VALIDATION", "false").lower() == "true"
        
        return SchemaConfig(
            schema_path=schema_path,
            enable_validation=enable_validation,
            strict_mode=strict_mode
        )
    
    def _init_pinecone_config(self) -> PineconeConfig:
        """Initialize Pinecone configuration from environment"""
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        
        if not api_key or not environment:
            raise ValueError(
                "Missing required Pinecone environment variables"
            )
        
        return PineconeConfig(api_key=api_key, environment=environment)
    
    def validate(self):
        """Comprehensive validation of all configuration components"""
        # Validate individual components
        self.metadata.validate()
        self.pinecone.validate()
        self.processing.validate()
        
        # Validate file paths
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Cross-component validation
        if (self.metadata.batch_size > 
            self.processing.max_validation_queue):
            raise ValueError(
                "Batch size cannot exceed maximum validation queue size"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format"""
        return {
            # Schema settings
            "SCHEMA_PATH": str(self.schema.schema_path),
            "ENABLE_VALIDATION": self.schema.enable_validation,
            "STRICT_MODE": self.schema.strict_mode,
            "CONFIDENCE_THRESHOLDS": self.schema.confidence_thresholds,
            
            # Metadata settings
            "NAMESPACE_STRATEGY": self.metadata.namespace_strategy,
            "METADATA_BATCH_SIZE": self.metadata.batch_size,
            "EMBEDDING_DIMENSION": self.metadata.embedding_dimension,
            "PREDICTIVE_FIELDS": self.metadata.predictive_fields,
            "FEATURE_CONFIG": self.metadata.feature_config,
            
            # Pinecone settings
            "PINECONE_API_KEY": self.pinecone.api_key,
            "PINECONE_ENVIRONMENT": self.pinecone.environment,
            "DOCUMENT_INDEX": self.pinecone.document_index_name,
            "METADATA_INDEX": self.pinecone.metadata_index_name,
            "METADATA_FIELDS": self.pinecone.metadata_fields,
            
            # Processing settings
            "CHUNK_SIZE": self.processing.chunk_size,
            "CHUNK_OVERLAP": self.processing.chunk_overlap,
            "MAX_RETRIES": self.processing.max_retries,
            "VALIDATION_WORKERS": self.processing.validation_workers,
            "SKIP_INVALID_CHUNKS": self.processing.skip_invalid_chunks,
            
            # File paths
            "DATA_DIR": str(self.data_dir),
            "EXAMPLES_PATH": str(self.examples_path)
        }

# Global configuration instance
_config: Optional[AppConfig] = None

def get_config() -> Dict[str, Any]:
    """Get application configuration with lazy initialization"""
    global _config
    
    if _config is None:
        try:
            _config = AppConfig()
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {str(e)}")
            raise
    
    return _config.to_dict()

@lru_cache(maxsize=1)
def load_schema() -> Dict[str, Any]:
    """Load and cache schema with validation"""
    global _config
    
    if _config is None:
        _config = AppConfig()
    
    return _config.schema.load_schema()