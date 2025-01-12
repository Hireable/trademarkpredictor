import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, conint, confloat
from jsonschema import validate, ValidationError as JsonSchemaError

from src.exceptions import ValidationError
from src.config.settings import get_settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class SimilarityLevel(Enum):
    """Enumeration for similarity assessment levels"""
    DISSIMILAR = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    IDENTICAL = 5

class ConfusionType(Enum):
    """Enumeration for types of confusion"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    BOTH = "both"
    NONE = "none"

class GoodServiceComparison(BaseModel):
    """Model for goods/services comparison data"""
    nice_class: conint(ge=1, le=45) = Field(..., description="NICE classification number")
    description: str = Field(..., description="Description of goods/services")
    similarity_level: conint(ge=1, le=5) = Field(..., description="Similarity level (1-5)")
    is_complementary: Optional[bool] = Field(None, description="Whether goods/services are complementary")
    is_competitive: Optional[bool] = Field(None, description="Whether goods/services are competitive")

class Mark(BaseModel):
    """Model for trademark information"""
    text: str = Field(..., description="Text of the mark")
    nice_classes: list[conint(ge=1, le=45)] = Field(..., description="NICE classes covered")
    owner: str = Field(..., description="Mark owner")
    application_number: Optional[str] = Field(None, description="Application number")
    registration_date: Optional[str] = Field(None, description="Registration date (DD/MM/YYYY)")

class SimilarityAssessment(BaseModel):
    """Model for mark similarity assessment"""
    visual: conint(ge=1, le=5) = Field(..., description="Visual similarity (1-5)")
    aural: conint(ge=1, le=5) = Field(..., description="Aural similarity (1-5)")
    conceptual: Optional[conint(ge=1, le=5)] = Field(None, description="Conceptual similarity (1-5)")
    overall: conint(ge=1, le=5) = Field(..., description="Overall similarity assessment (1-5)")

class TrademarkDecision(BaseModel):
    """
    Main model for trademark opposition decision data.
    This defines the expected structure of processed trademark decisions.
    """
    case_number: str = Field(..., description="Unique case identifier")
    decision_date: str = Field(..., description="Decision date (DD/MM/YYYY)")
    
    earlier_mark: Mark = Field(..., description="Details of the earlier mark")
    contested_mark: Mark = Field(..., description="Details of the contested mark")
    
    similarity_assessment: SimilarityAssessment = Field(
        ..., 
        description="Assessment of marks similarity"
    )
    
    goods_services_comparisons: list[GoodServiceComparison] = Field(
        ..., 
        description="Comparisons of goods/services",
        min_items=1
    )
    
    attention_level: conint(ge=1, le=5) = Field(
        ..., 
        description="Consumer attention level (1-5)"
    )
    
    distinctiveness: conint(ge=1, le=5) = Field(
        ..., 
        description="Distinctiveness of earlier mark (1-5)"
    )
    
    confusion_found: bool = Field(..., description="Whether confusion was found")
    confusion_type: Optional[ConfusionType] = Field(
        None, 
        description="Type of confusion if found"
    )
    
    key_findings: list[str] = Field(
        ..., 
        description="Key findings from the decision",
        min_items=1
    )

class SchemaValidator:
    """
    Handles validation of JSON data against our trademark decision schema.
    Provides both Pydantic model validation and JSON Schema validation.
    """
    
    def __init__(self):
        """Initialize the schema validator with configuration"""
        self.settings = get_settings()
        self._schema: Optional[Dict[str, Any]] = None

    @property
    def schema(self) -> Dict[str, Any]:
        """
        Lazy load and cache the JSON schema.
        
        Returns:
            Dict containing the JSON schema
            
        Raises:
            ValidationError: If schema file cannot be loaded
        """
        if self._schema is None:
            try:
                with open(self.settings.schema_path, 'r', encoding='utf-8') as f:
                    self._schema = json.load(f)
                logger.info(f"Loaded schema from {self.settings.schema_path}")
            except Exception as e:
                logger.error(f"Failed to load schema: {str(e)}")
                raise ValidationError(f"Schema loading failed: {str(e)}")
        return self._schema

    def validate_decision(
        self, 
        data: Union[Dict[str, Any], str]
    ) -> TrademarkDecision:
        """
        Validate trademark decision data using both JSON Schema and Pydantic model.
        
        Args:
            data: Decision data as dictionary or JSON string
            
        Returns:
            Validated TrademarkDecision model
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Convert string to dict if needed
            if isinstance(data, str):
                data = json.loads(data)

            # Validate against JSON Schema
            validate(instance=data, schema=self.schema)
            
            # Validate using Pydantic model
            validated_decision = TrademarkDecision(**data)
            
            logger.info(f"Successfully validated decision {data.get('case_number', 'unknown')}")
            return validated_decision
            
        except JsonSchemaError as e:
            logger.error(f"JSON Schema validation failed: {str(e)}")
            raise ValidationError(f"Schema validation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            raise ValidationError(f"Validation failed: {str(e)}")

    def validate_batch(
        self,
        decisions: list[Dict[str, Any]]
    ) -> list[TrademarkDecision]:
        """
        Validate a batch of trademark decisions.
        
        Args:
            decisions: List of decision data dictionaries
            
        Returns:
            List of validated TrademarkDecision models
            
        Raises:
            ValidationError: If any decision fails validation
        """
        validated_decisions = []
        errors = []

        for idx, decision in enumerate(decisions):
            try:
                validated = self.validate_decision(decision)
                validated_decisions.append(validated)
            except ValidationError as e:
                errors.append(f"Decision {idx}: {str(e)}")

        if errors:
            raise ValidationError(
                f"Batch validation failed with {len(errors)} errors: {'; '.join(errors)}"
            )

        return validated_decisions

# Create a global validator instance
validator = SchemaValidator()

def get_validator() -> SchemaValidator:
    """
    Get the global schema validator instance.
    
    Returns:
        Initialized SchemaValidator
    """
    return validator