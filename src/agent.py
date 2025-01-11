import json
import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Generator, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from datetime import datetime
import concurrent.futures
from collections import deque
import weakref
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import aiohttp
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from jsonschema import validate, ValidationError
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_config, load_schema, VertexAIConfig, ProcessingConfig
from .utils import (
    MetadataProcessor,
    VectorStoreManager,
    QueryEnhancer,
    init_components,
    BatchProcessor
)
from .exceptions import (
    ConfigurationError,
    ProcessingError,
    SchemaValidationError,
    MetadataError,
    StorageError,
    ValidationDetail,
    ErrorCode
)

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Enhanced statistics tracking for document processing"""
    # Document statistics
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    
    # Chunk statistics
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    
    # Validation statistics
    validation_errors: int = 0
    schema_violations: int = 0
    
    # Token statistics
    total_tokens: int = 0
    total_embeddings: int = 0
    
    # Metadata statistics
    enriched_metadata: int = 0
    failed_metadata: int = 0
    
    # Performance tracking
    start_time: datetime = field(default_factory=datetime.now)
    processing_durations: List[float] = field(default_factory=list)
    
    def add_duration(self, duration: float):
        """Add processing duration and update statistics"""
        self.processing_durations.append(duration)
    
    @property
    def average_duration(self) -> float:
        """Calculate average processing duration"""
        if not self.processing_durations:
            return 0.0
        return sum(self.processing_durations) / len(self.processing_durations)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        total = self.processed_documents + self.failed_documents
        return (self.processed_documents / total * 100) if total > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to detailed dictionary format"""
        return {
            'document_stats': {
                'total': self.total_documents,
                'processed': self.processed_documents,
                'failed': self.failed_documents,
                'success_rate': self.success_rate
            },
            'chunk_stats': {
                'total': self.total_chunks,
                'processed': self.processed_chunks,
                'failed': self.failed_chunks
            },
            'validation_stats': {
                'errors': self.validation_errors,
                'schema_violations': self.schema_violations
            },
            'metadata_stats': {
                'enriched': self.enriched_metadata,
                'failed': self.failed_metadata
            },
            'performance_stats': {
                'average_duration': self.average_duration,
                'total_tokens': self.total_tokens,
                'total_embeddings': self.total_embeddings
            }
        }

class ResourcePool:
    """
    Enhanced resource pool with support for metadata processing and validation.
    Manages heavy resources with efficient lifecycle management.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        pool_size: int = 3,
        cleanup_interval: int = 300
    ):
        """Initialize resource pool with configuration"""
        self.config = config
        self.pool_size = pool_size
        self.cleanup_interval = cleanup_interval
        self.metadata_processor_limit = pool_size
        
        # Initialize resource dictionaries with weak references
        self._embedding_models = weakref.WeakValueDictionary()
        self._gemini_models = weakref.WeakValueDictionary()
        self._metadata_processors = weakref.WeakValueDictionary()
        
        # Resource locks and tracking
        self._model_locks = {}
        self._cleanup_tasks = set()
        self._last_cleanup = datetime.now()
        
        # Start cleanup task
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._cleanup_resources()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cleanup task error: {str(e)}")
        
        task = asyncio.create_task(cleanup_loop())
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def get_embedding_model(self) -> SentenceTransformer:
        """Get an available embedding model with enhanced error handling"""
        async with self._get_model_lock('embedding'):
            # Try to find an available model
            for key, model in self._embedding_models.items():
                if not self._model_locks.get(f'embedding_{key}', False):
                    self._model_locks[f'embedding_{key}'] = True
                    return model
            
            # Create new model if pool isn't full
            if len(self._embedding_models) < self.pool_size:
                try:
                    # Update this line to use the correct model
                    model = SentenceTransformer('intfloat/multilingual-e5-large')
                    key = f"model_{len(self._embedding_models)}"
                    self._embedding_models[key] = model
                    self._model_locks[f'embedding_{key}'] = True
                    return model
                except Exception as e:
                    raise ResourceError(
                        message=f"Failed to create embedding model: {str(e)}",
                        error_code=ErrorCode.RESOURCE_UNAVAILABLE
                    )
            
            # Wait for an available model
            while True:
                await asyncio.sleep(0.1)
                for key, model in self._embedding_models.items():
                    if not self._model_locks.get(f'embedding_{key}', False):
                        self._model_locks[f'embedding_{key}'] = True
                        return model

    async def get_metadata_processor(self) -> MetadataProcessor:
        """Get or create a metadata processor"""
        if not self._metadata_processors:
            processor = MetadataProcessor(
                schema=self.config['SCHEMA'],
                confidence_thresholds=self.config['CONFIDENCE_THRESHOLDS'],
                enable_validation=self.config['ENABLE_VALIDATION']
            )
            self._metadata_processors['default'] = processor
            
        return self._metadata_processors['default']

    @asynccontextmanager
    async def _get_model_lock(self, model_type: str):
        """Context manager for model access with error handling"""
        try:
            yield
        except Exception as e:
            logger.error(f"Error accessing {model_type} model: {str(e)}")
            raise
        finally:
            # Release the model lock
            for key in self._model_locks:
                if key.startswith(f'{model_type}_'):
                    self._model_locks[key] = False

    async def _cleanup_resources(self):
        """Cleanup unused resources"""
        current_time = datetime.now()
        
        # Skip if cleanup interval hasn't elapsed
        if (current_time - self._last_cleanup).total_seconds() < self.cleanup_interval:
            return
            
        try:
            # Clean up unused models
            for key in list(self._embedding_models.keys()):
                if not self._model_locks.get(f'embedding_{key}', False):
                    del self._embedding_models[key]
            
            # Update cleanup timestamp
            self._last_cleanup = current_time
            
        except Exception as e:
            logger.error(f"Resource cleanup error: {str(e)}")

    async def cleanup(self):
        """Cleanup all resources"""
        try:
            # Cancel cleanup tasks
            for task in self._cleanup_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._cleanup_tasks:
                await asyncio.gather(
                    *self._cleanup_tasks,
                    return_exceptions=True
                )
            
            # Clear resource dictionaries
            self._embedding_models.clear()
            self._gemini_models.clear()
            self._metadata_processors.clear()
            
            logger.info("Successfully cleaned up all resources")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")
            raise

class ChunkProcessor:
    """
    Enhanced chunk processor with metadata enrichment and validation.
    Handles document chunking and processing with advanced features.
    """
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        metadata_processor: MetadataProcessor,
        max_concurrent_chunks: int = 5
    ):
        """Initialize with enhanced processing capabilities"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_processor = metadata_processor
        self.max_concurrent_chunks = max_concurrent_chunks
        self.processing_queue = asyncio.Queue()
        self.stats = ProcessingStats()
        
        # Initialize chunk tracking
        self._processed_chunks: Set[str] = set()
        self._failed_chunks: Set[str] = set()
        self._processing_times: Dict[str, float] = {}
        
    async def process_document(
        self,
        document_path: str,
        processor_func: callable,
        namespace: str
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process document with enhanced metadata handling and validation.
        
        Args:
            document_path: Path to the document
            processor_func: Async function to process each chunk
            namespace: Target namespace
            
        Yields:
            Processed chunk results with enriched metadata
        """
        start_time = datetime.now()
        self.stats.total_documents += 1
        
        try:
            # Start chunk processing workers
            workers = [
                asyncio.create_task(self._chunk_worker(
                    processor_func,
                    namespace
                ))
                for _ in range(self.max_concurrent_chunks)
            ]

            # Queue chunks for processing
            async for chunk in self._generate_chunks(document_path):
                await self.processing_queue.put({
                    'content': chunk,
                    'document_path': document_path,
                    'chunk_id': self._generate_chunk_id(),
                    'namespace': namespace
                })
                self.stats.total_chunks += 1

            # Signal workers to finish
            for _ in range(self.max_concurrent_chunks):
                await self.processing_queue.put(None)

            # Wait for all chunks to be processed and gather results
            results = await asyncio.gather(*workers, return_exceptions=True)
            
            # Process results and update statistics
            for result_batch in results:
                if isinstance(result_batch, Exception):
                    logger.error(f"Worker failed: {str(result_batch)}")
                    self.stats.failed_chunks += len(workers)
                    continue
                    
                for result in result_batch:
                    if result:
                        self.stats.processed_chunks += 1
                        if result.get('enriched_metadata'):
                            self.stats.enriched_metadata += 1
                        yield result
                    else:
                        self.stats.failed_chunks += 1

            # Update document statistics
            if self.stats.failed_chunks == 0:
                self.stats.processed_documents += 1
            else:
                self.stats.failed_documents += 1

        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}")
            self.stats.failed_documents += 1
            raise ProcessingError(
                message=f"Document processing failed: {str(e)}",
                error_code=ErrorCode.PROCESSING_FAILED,
                details={'document_path': document_path}
            )

        finally:
            # Record processing duration
            duration = (datetime.now() - start_time).total_seconds()
            self.stats.add_duration(duration)

    async def _chunk_worker(
        self,
        processor_func: callable,
        namespace: str
    ) -> List[Dict[str, Any]]:
        """Enhanced worker for processing chunks with metadata handling"""
        results = []
        while True:
            chunk_data = await self.processing_queue.get()
            if chunk_data is None:
                break
                
            try:
                # Process chunk with metadata enrichment
                start_time = datetime.now()
                
                result = await processor_func(
                    chunk_data['content'],
                    namespace=namespace,
                    chunk_id=chunk_data['chunk_id']
                )
                
                # Record processing time
                duration = (datetime.now() - start_time).total_seconds()
                self._processing_times[chunk_data['chunk_id']] = duration
                
                if result:
                    self._processed_chunks.add(chunk_data['chunk_id'])
                    results.append(result)
                else:
                    self._failed_chunks.add(chunk_data['chunk_id'])
                    
            except SchemaValidationError as e:
                logger.error(
                    f"Schema validation error in chunk {chunk_data['chunk_id']}: {str(e)}"
                )
                self.stats.schema_violations += 1
                self._failed_chunks.add(chunk_data['chunk_id'])
                results.append(None)
                
            except Exception as e:
                logger.error(
                    f"Error processing chunk {chunk_data['chunk_id']}: {str(e)}"
                )
                self._failed_chunks.add(chunk_data['chunk_id'])
                results.append(None)
                
            finally:
                self.processing_queue.task_done()
                
        return results

    async def _generate_chunks(
        self,
        document_path: str
    ) -> Generator[str, None, None]:
        """Generate chunks with enhanced text processing"""
        try:
            elements = partition_pdf(str(document_path))
            current_chunk = []
            current_length = 0
            
            for element in elements:
                # Clean and normalize text
                text = self._clean_text(str(element))
                words = text.split()
                
                for word in words:
                    if current_length + len(word) + 1 <= self.chunk_size:
                        current_chunk.append(word)
                        current_length += len(word) + 1
                    else:
                        # Yield current chunk if valid
                        chunk_text = ' '.join(current_chunk)
                        if self._is_valid_chunk(chunk_text):
                            yield chunk_text
                        
                        # Start new chunk with overlap
                        overlap_words = current_chunk[-(self.chunk_overlap // 5):]
                        current_chunk = overlap_words + [word]
                        current_length = sum(len(w) + 1 for w in current_chunk)
            
            # Yield final chunk if valid
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if self._is_valid_chunk(chunk_text):
                    yield chunk_text
                    
        except Exception as e:
            logger.error(f"Error generating chunks from {document_path}: {str(e)}")
            raise ProcessingError(
                message=f"Chunk generation failed: {str(e)}",
                error_code=ErrorCode.CHUNK_PROCESSING_FAILED
            )

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with legal document support"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors in legal documents
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Fix merged words
        text = re.sub(r'[\u2018\u2019]', "'", text)
        text = re.sub(r'[\u2018\u2019]', "'", text)
        text = re.sub(r'[\u2018\u2019\u02BC\u2032]', "'", text)
        
        # Normalize section and paragraph markers
        text = re.sub(r'§', 'Section ', text)
        text = re.sub(r'¶', 'Paragraph ', text)
        
        # Fix common legal abbreviations
        text = re.sub(r'(?<=\d)(?:st|nd|rd|th)\b', '', text)
        
        return text

    def _is_valid_chunk(self, text: str) -> bool:
        """Enhanced chunk validation with metadata requirements"""
        if len(text.strip()) < 50:
            return False
            
        # Check for actual word content
        word_ratio = len(re.findall(r'\b\w+\b', text)) / len(text.split())
        if word_ratio < 0.5:
            return False
            
        # Check for pure formatting content
        if re.match(r'^[\s\d.,-]+$', text):
            return False
            
        return True

    def _generate_chunk_id(self) -> str:
        """Generate unique chunk identifier with timestamp"""
        return f"chunk_{datetime.now().timestamp()}_{id(self)}"

class MetadataEnricher:
    """
    Handles metadata enrichment and validation for trademark documents.
    Supports hierarchical metadata processing and predictive field generation.
    """
    def __init__(
        self,
        schema: Dict[str, Any],
        confidence_thresholds: Dict[str, float],
        enable_validation: bool = True
    ):
        """Initialize with schema and configuration"""
        self.schema = schema
        self.confidence_thresholds = confidence_thresholds
        self.enable_validation = enable_validation
        self._setup_cache()

    def _setup_cache(self):
        """Initialize caching for performance optimization"""
        self.cache = {
            'validation': {},
            'predictions': {},
            'embeddings': {}
        }
        self.cache_stats = {
            'hits': 0,
            'misses': 0
        }

    async def enrich_metadata(
        self,
        base_metadata: Dict[str, Any],
        content: str,
        namespace: str,
        chunk_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrich metadata with predictive fields and validation.
        
        Args:
            base_metadata: Base metadata to enrich
            content: Content for prediction generation
            namespace: Target namespace
            chunk_id: Optional chunk identifier
            
        Returns:
            Enriched metadata with predictions
            
        Raises:
            MetadataError: If enrichment fails
            SchemaValidationError: If validation fails
        """
        try:
            # Start with base metadata copy
            enriched = base_metadata.copy()

            # Generate predictions
            predictions = await self._generate_predictions(content)
            enriched.update(predictions)

            # Add similarity assessments
            if 'party_info' in enriched:
                enriched['similarity_assessment'] = (
                    await self._generate_similarity_assessment(
                        enriched['party_info'],
                        content
                    )
                )

            # Add outcome prediction
            if 'outcome' not in enriched:
                enriched['outcome'] = await self._predict_outcome(
                    enriched,
                    content
                )

            # Validate enriched metadata
            if self.enable_validation:
                self._validate_metadata(enriched, namespace, chunk_id)

            return enriched

        except Exception as e:
            raise MetadataError(
                message=f"Metadata enrichment failed: {str(e)}",
                namespace=namespace,
                chunk_id=chunk_id
            )

    async def _generate_predictions(
        self,
        content: str
    ) -> Dict[str, Any]:
        """Generate predictive fields from content"""
        cache_key = hash(content)
        if cache_key in self.cache['predictions']:
            self.cache_stats['hits'] += 1
            return self.cache['predictions'][cache_key]

        self.cache_stats['misses'] += 1
        try:
            # Implement prediction logic here
            # Generate predictions aligned with the updated schema
            predictions = {
                'confidence_score': 0.85,
                'distinct': 4,  # Distinctiveness score (scale 1–5)
                'attention': 3  # Consumer attention level (scale 1–5)
            }

            self.cache['predictions'][cache_key] = predictions
            return predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return {}

class TrademarkCaseAgent:
    """
    Enhanced trademark case processing agent with comprehensive metadata handling.
    Orchestrates document processing, metadata enrichment, and vector storage.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        processing_config: Optional['ProcessingConfig'] = None,
        vertex_ai_config: Optional[VertexAIConfig] = None
    ):
        """Initialize agent with enhanced configuration"""
        # Initialize configuration
        self.config = config or get_config()
        self.processing_config = processing_config or ProcessingConfig()
        self.vertex_ai_config = vertex_ai_config or VertexAIConfig.from_env()
        
        # Initialize components
        self.resource_pool = ResourcePool(self.config)
        self.vector_store_manager, self.metadata_processor, self.query_enhancer = (
            init_components(self.config)
        )
        
        # Create metadata enricher
        self.metadata_enricher = MetadataEnricher(
            schema=self.config['SCHEMA'],
            confidence_thresholds=self.config['CONFIDENCE_THRESHOLDS'],
            enable_validation=self.config['ENABLE_VALIDATION']
        )
        
        # Initialize processing queues
        self.embedding_queue = asyncio.Queue(maxsize=100)
        self.storage_queue = asyncio.Queue(maxsize=100)
        
        # Initialize statistics
        self.stats = ProcessingStats()
        
        # Start background tasks
        self._background_tasks = set()
        self._start_background_workers()

    def _start_background_workers(self):
        """Start background workers for embedding and storage"""
        for _ in range(3):
            task = asyncio.create_task(self._embedding_worker())
            self._background_tasks.add(task)
        
        for _ in range(2):
            task = asyncio.create_task(self._storage_worker())
            self._background_tasks.add(task)

    async def process_pdf_streaming(
        self,
        pdf_path: str,
        namespace: Optional[str] = None
    ) -> None:
        """
        Process PDF document with enhanced streaming and metadata enrichment.
        
        Args:
            pdf_path: Path to PDF document
            namespace: Optional namespace override
            
        Raises:
            ProcessingError: If processing fails
        """
        if namespace is None:
            namespace = Path(pdf_path).stem
        
        try:
            # Initialize chunk processor
            chunk_processor = ChunkProcessor(
                chunk_size=self.processing_config.chunk_size,
                chunk_overlap=self.processing_config.chunk_overlap,
                metadata_processor=self.metadata_processor,
                max_concurrent_chunks=self.processing_config.validation_workers
            )
            
            # Process document chunks with metadata enrichment
            async for processed_chunk in chunk_processor.process_document(
                pdf_path,
                self._process_chunk,
                namespace
            ):
                # Queue for embedding generation
                await self.embedding_queue.put({
                    'text': processed_chunk['text'],
                    'metadata': processed_chunk['metadata'],
                    'namespace': namespace,
                    'chunk_id': processed_chunk['chunk_id']
                })
                
                # Update statistics
                self.stats.total_chunks += 1
            
            # Wait for processing completion
            await self.embedding_queue.join()
            await self.storage_queue.join()
            
            # Update document statistics
            self.stats.processed_documents += 1
            
            logger.info(
                f"Completed processing {pdf_path} - "
                f"Chunks: {chunk_processor.stats.processed_chunks}/"
                f"{chunk_processor.stats.total_chunks} "
                f"(Failed: {chunk_processor.stats.failed_chunks})"
            )

        except Exception as e:
            self.stats.failed_documents += 1
            raise ProcessingError(
                message=f"Failed to process {pdf_path}: {str(e)}",
                error_code=ErrorCode.PROCESSING_FAILED,
                details={'pdf_path': pdf_path}
            )

    async def _process_chunk(
        self,
        chunk: str,
        namespace: str,
        chunk_id: str
    ) -> Dict[str, Any]:
        """
        Process a single chunk with metadata enrichment and validation.
        
        Args:
            chunk: Text chunk to process
            namespace: Target namespace
            chunk_id: Chunk identifier
            
        Returns:
            Processed chunk with enriched metadata
        """
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(chunk)
            
            # Skip invalid chunks
            if not self._is_valid_chunk(cleaned_text):
                logger.debug(f"Skipping invalid chunk: {chunk_id}")
                return None
            
            # Extract initial metadata
            base_metadata = await self._extract_metadata(cleaned_text)
            
            # Enrich metadata
            enriched_metadata = await self.metadata_enricher.enrich_metadata(
                base_metadata,
                cleaned_text,
                namespace,
                chunk_id
            )
            
            # Validate final metadata
            if self.config['ENABLE_VALIDATION']:
                self.metadata_processor.validate_metadata(
                    enriched_metadata,
                    namespace,
                    chunk_id
                )
            
            return {
                'chunk_id': chunk_id,
                'text': cleaned_text,
                'metadata': enriched_metadata,
                'timestamp': datetime.now().isoformat()
            }

        except SchemaValidationError as e:
            logger.error(f"Validation error in chunk {chunk_id}: {str(e)}")
            self.stats.validation_errors += 1
            if self.config['STRICT_MODE']:
                raise
            return None

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            raise ProcessingError(
                message=f"Chunk processing error: {str(e)}",
                error_code=ErrorCode.CHUNK_PROCESSING_FAILED,
                chunk_id=chunk_id
            )

    async def _embedding_worker(self):
        """Enhanced embedding worker with error handling"""
        try:
            while True:
                try:
                    batch = await self.embedding_queue.get()
                    if batch is None:
                        break
                except asyncio.CancelledError:
                    logger.info("Embedding worker cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    
                try:
                    # Generate embeddings
                    model = await self.resource_pool.get_embedding_model()
                    embedding = model.encode(batch['text'])
                    
                    # Queue for storage
                    await self.storage_queue.put({
                        'embedding': embedding,
                        'metadata': batch['metadata'],
                        'namespace': batch['namespace'],
                        'chunk_id': batch['chunk_id']
                    })
                    
                    # Update statistics
                    self.stats.total_embeddings += 1
                    
                except Exception as e:
                    logger.error(f"Embedding generation failed: {str(e)}")
                    self.stats.failed_chunks += 1
                    
                finally:
                    self.embedding_queue.task_done()
                    
        except asyncio.CancelledError:
            logger.info("Embedding worker cancelled")

    async def _storage_worker(self):
        """Enhanced storage worker with metadata handling"""
        try:
            while True:
                batch = await self.storage_queue.get()
                if batch is None:
                    break
                    
                try:
                    # Store vectors with metadata
                    await self.vector_store_manager.upsert_with_metadata(
                        vectors=[{
                            'id': batch['chunk_id'],
                            'values': batch['embedding'].tolist(),
                            'metadata': batch['metadata']
                        }],
                        namespace=batch['namespace']
                    )
                    
                    # Update statistics
                    self.stats.processed_chunks += 1
                    
                except Exception as e:
                    logger.error(f"Vector storage failed: {str(e)}")
                    self.stats.failed_chunks += 1
                    
                finally:
                    self.storage_queue.task_done()
                    
        except asyncio.CancelledError:
            logger.info("Storage worker cancelled")

    async def cleanup(self):
        """Enhanced cleanup with comprehensive resource management"""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(
                    *self._background_tasks,
                    return_exceptions=True
                )
            
            # Signal workers to stop
            for _ in range(3):
                await self.embedding_queue.put(None)
            for _ in range(2):
                await self.storage_queue.put(None)
            
            # Clean up resources
            await self.resource_pool.cleanup()
            
            logger.info("Successfully cleaned up agent resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self.cleanup()

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning for legal documents"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors in legal documents
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[\u2018\u2019]', "'", text)
        
        # Normalize legal markers and abbreviations
        text = re.sub(r'§', 'Section ', text)
        text = re.sub(r'¶', 'Paragraph ', text)
        text = re.sub(r'(?<=\d)(?:st|nd|rd|th)\b', '', text)
        
        return text

    def _is_valid_chunk(self, text: str) -> bool:
        """Enhanced chunk validation"""
        if len(text.strip()) < 50:
            return False
            
        word_ratio = len(re.findall(r'\b\w+\b', text)) / len(text.split())
        if word_ratio < 0.5:
            return False
            
        if re.match(r'^[\s\d.,-]+$', text):
            return False
            
        return True

    async def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract initial metadata from text"""
        try:
            # Generate extraction prompt
            prompt = self._generate_extraction_prompt(text)
            
            # Process with Vertex AI
            response = await self._extract_with_vertex_ai(prompt)
            
            # Parse and validate response
            metadata = self._parse_extraction_response(response)
            
            return metadata
            
        except Exception as e:
            raise MetadataError(
                message=f"Metadata extraction failed: {str(e)}",
                error_code=ErrorCode.METADATA_GENERATION_FAILED
            )

    def _generate_extraction_prompt(self, text: str) -> str:
        """Generate optimized prompt for metadata extraction"""
        return f"""Analyze the following trademark case text and extract key information:

Text: {text}

Please structure the response as JSON with the following components:
- case_metadata (reference number, dates, jurisdiction)
- party_info (marks, names, market presence)
- commercial_context (specifications, classifications)
- distinctiveness (inherent and acquired)
- outcome (if present)

Return only the JSON structure without additional commentary."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _extract_with_vertex_ai(self, prompt: str) -> Dict[str, Any]:
        """Extract information using Vertex AI with retry logic"""
        try:
            # Initialize Vertex AI model
            model = await self.resource_pool.get_gemini_model()
            
            # Generate response
            response = await model.generate_content(prompt)
            
            # Parse and validate response
            try:
                extracted_data = json.loads(response.text)
            except json.JSONDecodeError:
                raise MetadataError("Failed to parse model response as JSON")
                
            return extracted_data
            
        except Exception as e:
            logger.error(f"Vertex AI extraction failed: {str(e)}")
            raise

    def _parse_extraction_response(
        self,
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse and validate extraction response"""
        try:
            # Validate response structure
            required_fields = [
                'case_metadata',
                'party_info',
                'commercial_context'
            ]
            
            missing_fields = [
                field for field in required_fields
                if field not in response
            ]
            
            if missing_fields:
                raise MetadataError(
                    f"Missing required fields: {missing_fields}"
                )
            
            return response
            
        except Exception as e:
            raise MetadataError(f"Response parsing failed: {str(e)}")