import json
import logging
import re
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import aiohttp
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from jsonschema import validate, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

from src.config import get_config, load_schema, VertexAIConfig
from src.utils import get_pinecone_index, upsert_chunks, delete_namespace
from src.exceptions import (
    ConfigurationError,
    ProcessingError,
    ValidationError,
    ResourceError
)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration settings for document processing"""
    batch_size: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retries: int = 3
    wipo_confidence_threshold: float = 0.85
    embedding_dimension: int = 768

class ResourceManager:
    """Manages initialization and cleanup of heavy resources"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._embedding_model = None
        self._gemini_model = None

    @property
    @lru_cache(maxsize=1)
    def embedding_model(self):
        """Lazy loading of embedding model"""
        if not self._embedding_model:
            self._embedding_model = SentenceTransformer(
                self.config["EMBEDDING_MODEL_NAME"]
            )
        return self._embedding_model

    @property
    @lru_cache(maxsize=1)
    def gemini_model(self):
        """Lazy loading of Gemini model"""
        if not self._gemini_model:
            vertexai.init(project=self.config["VERTEX_PROJECT_ID"], location=self.config["VERTEX_LOCATION"])
            self._gemini_model = GenerativeModel("gemini-1.5-pro-002")
        return self._gemini_model

    async def cleanup(self):
        """Cleanup resources"""
        if self._embedding_model:
            pass

class TrademarkCaseAgent:
    """
    Enhanced agent for processing trademark cases with improved resource management,
    error handling, and concurrent processing capabilities.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        processing_config: Optional[ProcessingConfig] = None,
        vertex_ai_config: Optional[VertexAIConfig] = None
    ):
        """
        Initialize the TrademarkCaseAgent with configuration and resources.

        Args:
            config: Optional custom configuration dictionary
            processing_config: Optional custom processing configuration

        Raises:
            ConfigurationError: If required configuration is missing or invalid
        """
        self.config = config or self._load_and_validate_config()
        self.processing_config = processing_config or ProcessingConfig()
        self.vertex_ai_config = vertex_ai_config or VertexAIConfig.from_env()

        # Initialize resource manager
        self.resources = ResourceManager(self.config)

        # Load and validate schema and few-shot examples
        self.schema = self._load_schema()
        self.few_shot_examples = self._load_few_shot_examples()

        # Initialize vector store connections
        self.index = self._initialize_vector_store()

        # Initialize asyncio session for concurrent operations
        self.session = None

        self.generation_config = {
            "max_output_tokens": 2053,
            "temperature": 0.2,
            "top_p": 0.96,
            "response_mime_type": "application/json",
        }
        self.safety_settings = {
            SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        }

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate configuration settings"""
        config = get_config()
        required_keys = [
            "PINECONE_INDEX_NAME",
            "EMBEDDING_MODEL_NAME",
            "WIPO_INDEX_NAME",
            "SCHEMA_PATH",
            "EXAMPLES_PATH"
        ]

        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ConfigurationError(
                f"Missing required configuration keys: {missing_keys}"
            )

        return config
    
    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples from examples.json"""
        examples_path = Path(self.config["EXAMPLES_PATH"])
        if not examples_path.exists():
            raise FileNotFoundError(f"Few-shot examples file not found: {examples_path}")
        try:
            # Specify encoding explicitly
            with open(examples_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except UnicodeDecodeError as e:
            raise ConfigurationError(f"Encoding issue in few-shot examples file: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load few-shot examples: {str(e)}")

    def _load_schema(self) -> Dict[str, Any]:
        """Load and validate JSON schema"""
        try:
            return load_schema(self.config["SCHEMA_PATH"])
        except Exception as e:
            raise ConfigurationError(f"Failed to load schema: {str(e)}")

    def _initialize_vector_store(self):
        """Initialize connection to vector store"""
        try:
            return get_pinecone_index(
                self.config["PINECONE_INDEX_NAME"],
                self.processing_config.embedding_dimension
            )
        except Exception as e:
            raise ResourceError(f"Failed to initialize vector store: {str(e)}")

    @asynccontextmanager
    async def session_context(self):
        """Context manager for aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        try:
            yield self.session
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    async def process_pdf_streaming(
        self,
        pdf_path: str,
        namespace: Optional[str] = None
    ) -> None:
        """
        Process PDF document using streaming to handle large files efficiently.

        Args:
            pdf_path: Path to PDF file
            namespace: Optional namespace for vector store

        Raises:
            ProcessingError: If PDF processing fails
        """
        if namespace is None:
            namespace = Path(pdf_path).stem

        # Clean up existing namespace
        await self._cleanup_namespace(namespace)

        try:
            # Process PDF in chunks
            async for chunk in self._stream_pdf_chunks(pdf_path):
                embeddings = await self._generate_embeddings(chunk)
                await self._store_chunk_embeddings(
                    chunk,
                    embeddings,
                    namespace,
                    pdf_path
                )

            logger.info(
                f"Successfully processed PDF: {pdf_path} into namespace: {namespace}"
            )

        except Exception as e:
            raise ProcessingError(f"Failed to process PDF {pdf_path}: {str(e)}")

    async def _cleanup_namespace(self, namespace: str) -> None:
        """Clean up existing namespace in vector store"""
        try:
            await delete_namespace(self.index, namespace)
        except Exception as e:
            logger.warning(f"Failed to cleanup namespace {namespace}: {str(e)}")

    async def _stream_pdf_chunks(self, pdf_path: str):
        """Stream PDF content in chunks"""
        elements = partition_pdf(filename=pdf_path)
        current_chunk = []
        current_size = 0

        for element in elements:
            element_text = str(element)
            current_chunk.append(element_text)
            current_size += len(element_text)

            if current_size >= self.processing_config.chunk_size:
                yield "\n\n".join(current_chunk)
                current_chunk = []
                current_size = 0

        if current_chunk:
            yield "\n\n".join(current_chunk)

    async def _generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text chunk"""
        return self.resources.embedding_model.encode(text).tolist()

    async def _store_chunk_embeddings(
        self,
        chunk: str,
        embedding: List[float],
        namespace: str,
        source_file: str
    ) -> None:
        """Store chunk embeddings"""
        chunk_id = f"{namespace}-{hash(chunk)}"
        metadata = {
            "text": chunk,
            "filename": Path(source_file).name,
            "namespace": namespace,
        }

        upsert_chunks(
            self.index,
            [{
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace
        )

    def create_prompt_with_examples(self, case_text):
        """
        Constructs a prompt for Vertex AI, including few-shot examples.
        """
        system_instruction = "You are a specialised legal AI assistant focused on analysing trademark opposition decisions in PDF and text format, and extracting structured data. You may only return a valid JSON object according to the provided response schema."
        
        prompt = f"{system_instruction}\n\nHere are your critical instructions and guidelines:\n\n"
        
        prompt += """1. Core Case Information 
        a) Automatically identify the relevant section 5(2)(b) assessment 
        b) Skip standard case law recitations 
        c) Focus on the hearing officer's direct analysis

        2. Mark & Party Information 
        a) Extract mark details from the document opening 
        b) For multiple marks, select the most relevant word mark for comparison 
        c) Avoid including stylised or device marks unless necessary

        3. Goods & Services Analysis 
        a) Identify broader category comparisons rather than specific items 
        b) Focus on mainstream goods/services over niche items 
        c) Limit to 3 most representative comparisons per case 
        d) Use exact NICE/WIPO classifications from the reference database

        4. Similarity Assessment 
        a) Record explicit similarity findings (visual, aural, conceptual) using the 1-5 scale: 
        i) 1 = Dissimilar 
        ii) 2 = Low 
        iii) 3 = Medium 
        iv) 4 = High 
        v) 5 = Identical 
        b) When conceptual comparison is not possible, set value to null 
        c) For identical goods, mark competition/complementary fields as null 
        d) Extract clear findings on attention level and distinctiveness using same 1-5 scale

        5. Comparison Framework 
        a) Focus on the broadest, most relevant comparisons 
        b) Skip repetitive or highly specific comparisons 
        c) Look for explicit statements about: 
        i) Nature of goods/services 
        ii) Trade channels 
        iii) Method of use 
        iv) Purpose 
        v) Competition/complementary relationship

        6. Decision Outcome 
        a) Extract clear findings on likelihood of confusion 
        b) Identify direct vs indirect confusion where specified 
        c) Note scope of confusion (which goods/services affected)

        7. Output Guidelines: 
        a) Adhere strictly to the provided JSON schema 
        b) Use only enumerated values for categorical fields 
        c) Mark fields as null when information is unavailable or not applicable 
        d) Maintain consistent assessment criteria across cases 
        e) Format dates as DD/MM/YYYY 
        f) Include NICE classes 1-45 only 
        g) Use numerical scale 1-5 for all similarity and strength assessments 
        h) Boolean values for complementary/competitive assessments

        Focus on accuracy and consistency over comprehensiveness. Extract clear findings rather than making assumptions about ambiguous information.

        Here are a few examples to guide the output format and structure:\n"""
        
        for example in self.few_shot_examples:
            prompt += f"\ninput: {example['input']}\noutput: {json.dumps(example['output'])}\n"

        prompt += f"\ninput: {case_text}\noutput:"
        
        return prompt

    async def generate_json_output(self, text_content: str) -> Optional[Dict]:
        """
        Generate structured JSON output from text content using Vertex AI.
    
        Args:
        text_content: The text content to process
        
        Returns:
        Dictionary containing the structured data, or None if generation fails
        """
        try:
        # Initialize the model with proper configuration
            model = GenerativeModel(
            "gemini-pro",
            generation_config={"temperature": 0.2}  # Low temperature for consistent output
        )
        
        # Create the prompt with the schema and example
        prompt = self._create_json_prompt(text_content)
        
        # Generate the response
        response = await model.generate_content(prompt)
        
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            logging.error("Invalid response format from Vertex AI")
            return None
            
        # Extract the JSON string from the response
        json_str = response.candidates[0].content
        
        # Parse and validate the JSON
        try:
            json_data = json.loads(json_str)
            # Add validation against schema if needed
            return json_data
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return None
            
        except Exception as e:
            logging.error(f"Error during JSON generation or validation: {str(e)}")
        return None

    def _extract_and_validate_json(self, result: str) -> Dict[str, Any]:
        """Extract and validate JSON from LLM response"""
        try:
            # Directly attempt to parse the response as JSON
            json_output = json.loads(result)

            # Validate against schema
            validate(instance=json_output, schema=self.schema)
            return json_output
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}")
            logger.error(f"Problematic text: {result}")
            raise ValidationError("Invalid JSON format in response")
        except ValidationError as e:
            logger.error(f"JSON schema validation error: {e}")
            raise ValidationError(f"JSON does not conform to the schema: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources when done"""
        await self.resources.cleanup()
        if self.session:
            await self.session.close()