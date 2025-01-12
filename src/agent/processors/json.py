"""
JSON processing utilities for generation, validation, and schema management.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.generative_models._generative_models import GenerateContentResponse

from src.exceptions import ProcessingError, ValidationError
from src.config.schema import get_validator, TrademarkDecision
from src.config.settings import get_settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for JSON generation."""
    max_output_tokens: int = 2053
    temperature: float = 0.2
    top_p: float = 0.96
    response_mime_type: str = "application/json"
    max_retries: int = 3

@dataclass
class Example:
    """Represents a few-shot example for model prompting."""
    input_text: str
    output_json: Dict[str, Any]

    def format(self) -> str:
        """Format example for inclusion in prompt."""
        return f"input: {self.input_text}\noutput: {json.dumps(self.output_json)}\n"

class JsonProcessor:
    """Handles JSON generation, validation, and schema management."""

    def __init__(
        self,
        examples_path: Path,
        generation_config: Optional[GenerationConfig] = None,
        vertex_project_id: Optional[str] = None,
        vertex_location: Optional[str] = "europe-west2"
    ):
        """
        Initialize the JSON processor.

        Args:
            examples_path: Path to few-shot examples file
            generation_config: Optional generation configuration
            vertex_project_id: Optional Vertex AI project ID
            vertex_location: Optional Vertex AI location

        Raises:
            ProcessingError: If initialization fails
        """
        self.validator = get_validator()
        self.examples = self._load_examples(examples_path)
        self.generation_config = generation_config or GenerationConfig()
        
        # Initialize Vertex AI if project ID provided
        if vertex_project_id:
            try:
                vertexai.init(
                    project=vertex_project_id,
                    location=vertex_location
                )
                self._model = GenerativeModel("gemini-1.5-pro-002")
                logger.info("Initialized Vertex AI model")
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI: {str(e)}")
                raise ProcessingError(f"Failed to initialize Vertex AI: {str(e)}")
        else:
            self._model = None

    def _load_examples(self, examples_path: Path) -> List[Example]:
        """Load few-shot examples."""
        try:
            if not examples_path.exists():
                raise FileNotFoundError(f"Examples file not found: {examples_path}")
            
            with open(examples_path, "r", encoding="utf-8") as f:
                raw_examples = json.load(f)
            
            examples = [
                Example(
                    input_text=example["input"],
                    output_json=example["output"]
                )
                for example in raw_examples
            ]
            
            logger.info(f"Loaded {len(examples)} few-shot examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load examples: {str(e)}")
            raise ProcessingError(f"Failed to load examples: {str(e)}")

    def create_prompt(self, text: str) -> str:
        """Create prompt for JSON generation including examples."""
        system_instruction = (
            "You are a specialized legal AI assistant focused on analyzing trademark "
            "opposition decisions and extracting structured data. You must only return "
            "a valid JSON object according to the provided response schema."
        )
        
        prompt = f"{system_instruction}\n\n"
        
        # Add critical instructions
        prompt += (
            "Critical Instructions:\n"
            "1. Core Case Information:\n"
            "   - Focus on section 5(2)(b) assessment\n"
            "   - Skip standard case law recitations\n"
            "   - Extract direct analysis by hearing officer\n\n"
            "2. Mark & Party Information:\n"
            "   - Extract details from document opening\n"
            "   - For multiple marks, select most relevant word mark\n"
            "   - Include stylized/device marks only if necessary\n\n"
            "3. Goods & Services Analysis:\n"
            "   - Focus on broader categories over specific items\n"
            "   - Limit to 3 most representative comparisons\n"
            "   - Use exact NICE classifications\n\n"
            "4. Similarity Assessment:\n"
            "   - Use 1-5 scale for similarity (1=Dissimilar, 5=Identical)\n"
            "   - Set conceptual value to null if comparison impossible\n"
            "   - Mark competition/complementary as null for identical goods\n\n"
            "5. Output Requirements:\n"
            "   - Return ONLY valid JSON\n"
            "   - Follow schema exactly\n"
            "   - Use null for missing information\n"
            "   - Format dates as DD/MM/YYYY\n"
            "   - Use boolean values for competitive/complementary fields\n\n"
        )
        
        # Add examples
        if self.examples:
            prompt += "Examples:\n"
            for example in self.examples:
                prompt += example.format()
        
        # Add input text
        prompt += f"\ninput: {text}\noutput:"
        
        return prompt

    async def generate_json(self, text: str) -> TrademarkDecision:
        """
        Generate JSON from text using Vertex AI.

        Args:
            text: Input text to process

        Returns:
            Validated TrademarkDecision object

        Raises:
            ProcessingError: If generation fails
            ValidationError: If generated JSON is invalid
        """
        if not self._model:
            raise ProcessingError("Vertex AI model not initialized")

        try:
            prompt = self.create_prompt(text)
            
            response = await self._model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": self.generation_config.max_output_tokens,
                    "temperature": self.generation_config.temperature,
                    "top_p": self.generation_config.top_p
                }
            )
            
            if not response or not response.text:
                raise ProcessingError("Empty response from Vertex AI")
            
            # Parse and validate JSON
            try:
                json_output = json.loads(response.text)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in response: {str(e)}")
                logger.error(f"Response text: {response.text}")
                raise ValidationError(f"Invalid JSON in response: {str(e)}")
            
            # Validate and convert to TrademarkDecision
            validated_decision = self.validator.validate_decision(json_output)
            logger.info(f"Successfully generated and validated JSON for case {validated_decision.case_number}")
            
            return validated_decision
            
        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            logger.error(f"JSON generation failed: {str(e)}")
            raise ProcessingError(f"JSON generation failed: {str(e)}")

    def merge_decisions(self, decisions: List[Dict[str, Any]]) -> TrademarkDecision:
        """
        Merge multiple trademark decisions.

        Args:
            decisions: List of decision dictionaries to merge

        Returns:
            Merged and validated TrademarkDecision

        Raises:
            ValidationError: If merged result is invalid
        """
        if not decisions:
            raise ValueError("Cannot merge empty list of decisions")

        try:
            # Start with first decision
            result = decisions[0].copy()

            # Merge subsequent decisions
            for decision in decisions[1:]:
                self._deep_merge(result, decision)

            # Validate merged result
            validated_decision = self.validator.validate_decision(result)
            logger.info(f"Successfully merged {len(decisions)} decisions")
            
            return validated_decision

        except Exception as e:
            logger.error(f"Decision merge failed: {str(e)}")
            raise ValidationError(f"Failed to merge decisions: {str(e)}")

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively merge source into target."""
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    self._deep_merge(target[key], value)
                elif isinstance(target[key], list) and isinstance(value, list):
                    target[key].extend(value)
                else:
                    # Prefer non-null values
                    if target[key] is None:
                        target[key] = value
            else:
                target[key] = value