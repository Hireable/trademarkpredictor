from typing import Optional, Dict, Any, List
from datetime import datetime
import traceback
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ErrorCode:
    """Comprehensive error codes for trademark processing system"""
    
    # Configuration errors (1xxx)
    CONFIG_MISSING = 1001
    CONFIG_INVALID = 1002
    ENV_VAR_MISSING = 1003
    ENV_VAR_INVALID = 1004
    SCHEMA_MISSING = 1005
    SCHEMA_INVALID = 1006
    
    # Processing errors (2xxx)
    PROCESSING_FAILED = 2001
    PDF_PARSING_FAILED = 2002
    TEXT_EXTRACTION_FAILED = 2003
    CHUNK_PROCESSING_FAILED = 2004
    BATCH_PROCESSING_FAILED = 2005
    WORKER_FAILED = 2006
    
    # Validation errors (3xxx)
    SCHEMA_VALIDATION_FAILED = 3001
    DATA_FORMAT_INVALID = 3002
    REQUIRED_FIELD_MISSING = 3003
    FIELD_TYPE_INVALID = 3004
    ENUM_VALUE_INVALID = 3005
    ARRAY_VALIDATION_FAILED = 3006
    
    # Metadata errors (4xxx)
    METADATA_GENERATION_FAILED = 4001
    PREDICTIVE_FIELD_ERROR = 4002
    CONFIDENCE_THRESHOLD_ERROR = 4003
    METADATA_VALIDATION_FAILED = 4004
    NAMESPACE_ERROR = 4005
    ENRICHMENT_FAILED = 4006
    
    # Resource errors (5xxx)
    RESOURCE_UNAVAILABLE = 5001
    CONNECTION_FAILED = 5002
    API_ERROR = 5003
    RATE_LIMIT_EXCEEDED = 5004
    INDEX_ERROR = 5005
    
    # Storage errors (6xxx)
    VECTOR_STORE_ERROR = 6001
    INDEX_NOT_FOUND = 6002
    UPSERT_FAILED = 6003
    QUERY_FAILED = 6004
    BATCH_UPSERT_FAILED = 6005

class ValidationDetail:
    """Detailed information about a validation failure"""
    
    def __init__(
        self,
        field_path: str,
        error_type: str,
        expected_type: Optional[str] = None,
        received_value: Any = None,
        constraints: Optional[Dict[str, Any]] = None
    ):
        self.field_path = field_path
        self.error_type = error_type
        self.expected_type = expected_type
        self.received_value = received_value
        self.constraints = constraints or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation detail to dictionary format"""
        return {
            'field_path': self.field_path,
            'error_type': self.error_type,
            'expected_type': self.expected_type,
            'received_value': str(self.received_value),
            'constraints': self.constraints,
            'timestamp': self.timestamp.isoformat()
        }

class ErrorContext:
    """Enhanced error context with support for validation details"""
    
    def __init__(
        self,
        error_code: int,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        validation_details: Optional[List[ValidationDetail]] = None,
        namespace: Optional[str] = None,
        chunk_id: Optional[str] = None
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.validation_details = validation_details or []
        self.namespace = namespace
        self.chunk_id = chunk_id
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
    
    def add_validation_detail(self, validation_detail: ValidationDetail):
        """Add a validation detail to the error context"""
        self.validation_details.append(validation_detail)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to detailed dictionary format"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'validation_details': [
                detail.to_dict() for detail in self.validation_details
            ],
            'namespace': self.namespace,
            'chunk_id': self.chunk_id,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback
        }
    
    def to_json(self) -> str:
        """Convert error context to formatted JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def log_error(self, logger: logging.Logger = logger):
        """Enhanced error logging with validation details"""
        log_message = (
            f"Error {self.error_code}: {self.message}\n"
            f"Namespace: {self.namespace}\n"
            f"Chunk ID: {self.chunk_id}"
        )
        
        if self.validation_details:
            log_message += "\nValidation Details:"
            for detail in self.validation_details:
                log_message += f"\n - {detail.field_path}: {detail.error_type}"
        
        logger.error(
            log_message,
            extra={
                'error_context': self.to_dict(),
                'error_code': self.error_code
            }
        )

class TrademarkAgentError(Exception):
    """Enhanced base exception class with validation support"""
    
    def __init__(
        self,
        message: str,
        error_code: int,
        details: Optional[Dict[str, Any]] = None,
        validation_details: Optional[List[ValidationDetail]] = None,
        namespace: Optional[str] = None,
        chunk_id: Optional[str] = None
    ):
        super().__init__(message)
        self.context = ErrorContext(
            error_code=error_code,
            message=message,
            details=details,
            validation_details=validation_details,
            namespace=namespace,
            chunk_id=chunk_id
        )
    
    def __str__(self) -> str:
        """Enhanced string representation with validation details"""
        base_str = f"[{self.context.error_code}] {self.context.message}"
        if self.context.validation_details:
            base_str += "\nValidation Errors:"
            for detail in self.context.validation_details:
                base_str += f"\n - {detail.field_path}: {detail.error_type}"
        return base_str
    
    def log(self, logger: logging.Logger = logger):
        """Log error with enhanced context"""
        self.context.log_error(logger)

class SchemaValidationError(TrademarkAgentError):
    """Enhanced schema validation error with detailed reporting"""
    
    def __init__(
        self,
        message: str,
        field_path: str,
        expected_type: str,
        received_value: Any,
        constraints: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        chunk_id: Optional[str] = None
    ):
        validation_detail = ValidationDetail(
            field_path=field_path,
            error_type="schema_validation",
            expected_type=expected_type,
            received_value=received_value,
            constraints=constraints
        )
        
        super().__init__(
            message=message,
            error_code=ErrorCode.SCHEMA_VALIDATION_FAILED,
            validation_details=[validation_detail],
            namespace=namespace,
            chunk_id=chunk_id
        )

class MetadataError(TrademarkAgentError):
    """Error handling for metadata processing and enrichment"""
    
    def __init__(
        self,
        message: str,
        error_code: int = ErrorCode.METADATA_GENERATION_FAILED,
        field_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        chunk_id: Optional[str] = None
    ):
        if field_path:
            validation_detail = ValidationDetail(
                field_path=field_path,
                error_type="metadata_generation"
            )
            validation_details = [validation_detail]
        else:
            validation_details = None
            
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            validation_details=validation_details,
            namespace=namespace,
            chunk_id=chunk_id
        )

class ProcessingError(TrademarkAgentError):
    """Enhanced processing error with batch and worker support"""
    
    def __init__(
        self,
        message: str,
        error_code: int = ErrorCode.PROCESSING_FAILED,
        batch_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        failed_items: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        chunk_id: Optional[str] = None
    ):
        details = {
            'batch_id': batch_id,
            'worker_id': worker_id,
            'failed_items': failed_items or []
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            namespace=namespace,
            chunk_id=chunk_id
        )

class StorageError(TrademarkAgentError):
    """Enhanced storage error with vector store specifics"""
    
    def __init__(
        self,
        message: str,
        error_code: int = ErrorCode.VECTOR_STORE_ERROR,
        index_name: Optional[str] = None,
        operation: Optional[str] = None,
        affected_vectors: Optional[List[str]] = None,
        namespace: Optional[str] = None
    ):
        details = {
            'index_name': index_name,
            'operation': operation,
            'affected_vectors': affected_vectors or []
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            namespace=namespace
        )

class ConfigurationError(TrademarkAgentError):
    """Enhanced configuration error with schema support"""
    
    def __init__(
        self,
        message: str,
        error_code: int = ErrorCode.CONFIG_INVALID,
        config_path: Optional[str] = None,
        invalid_settings: Optional[Dict[str, Any]] = None
    ):
        details = {
            'config_path': config_path,
            'invalid_settings': invalid_settings or {}
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details
        )
    
    @classmethod
    def schema_error(
        cls,
        schema_path: str,
        error_details: Dict[str, Any]
    ) -> 'ConfigurationError':
        """Create specific error for schema configuration issues"""
        return cls(
            message=f"Invalid schema configuration: {schema_path}",
            error_code=ErrorCode.SCHEMA_INVALID,
            config_path=schema_path,
            invalid_settings=error_details
        )