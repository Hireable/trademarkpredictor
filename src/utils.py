import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
import json
from contextlib import asynccontextmanager
import numpy as np
from collections import deque
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, Index
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from jsonschema import validate, ValidationError

from .exceptions import (
    SchemaValidationError,
    MetadataError,
    StorageError,
    ValidationDetail,
    ErrorCode
)

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class MetadataProcessor:
    """
    Processes and enriches metadata according to schema requirements.
    Handles validation, enrichment, and transformation of metadata.
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
        self._setup_validation_cache()

    def _setup_validation_cache(self):
        """Initialize LRU cache for validation results"""
        self.validation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    async def process_metadata(
        self,
        metadata: Dict[str, Any],
        namespace: str,
        chunk_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process and enrich metadata with validation and predictive fields.
        
        Args:
            metadata: Raw metadata to process
            namespace: Target namespace
            chunk_id: Optional chunk identifier
            
        Returns:
            Enriched and validated metadata
            
        Raises:
            SchemaValidationError: If validation fails
            MetadataError: If enrichment fails
        """
        try:
            # Validate base metadata
            if self.enable_validation:
                self._validate_metadata(metadata, namespace, chunk_id)

            # Enrich with predictive fields
            enriched = await self._enrich_metadata(metadata)

            # Validate enriched metadata
            if self.enable_validation:
                self._validate_metadata(enriched, namespace, chunk_id)

            return enriched

        except ValidationError as e:
            raise SchemaValidationError(
                message="Metadata validation failed",
                field_path=str(e.path),
                expected_type=str(e.validator),
                received_value=e.instance,
                namespace=namespace,
                chunk_id=chunk_id
            )
        except Exception as e:
            raise MetadataError(
                message=f"Metadata processing failed: {str(e)}",
                namespace=namespace,
                chunk_id=chunk_id
            )

    def _validate_metadata(
        self,
        metadata: Dict[str, Any],
        namespace: str,
        chunk_id: Optional[str] = None
    ):
        """Validate metadata against schema with caching"""
        # Check cache first
        cache_key = self._get_cache_key(metadata)
        if cache_key in self.validation_cache:
            self.cache_hits += 1
            if not self.validation_cache[cache_key]:
                raise self.validation_cache[cache_key]
            return

        self.cache_misses += 1
        try:
            validate(instance=metadata, schema=self.schema)
            self.validation_cache[cache_key] = True
        except ValidationError as e:
            error = SchemaValidationError(
                message="Schema validation failed",
                field_path=str(e.path),
                expected_type=str(e.validator),
                received_value=e.instance,
                namespace=namespace,
                chunk_id=chunk_id
            )
            self.validation_cache[cache_key] = error
            raise error

    async def _enrich_metadata(
        self,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich metadata with predictive fields and derived data.
        Handles the generation and validation of enriched fields.
        """
        enriched = metadata.copy()

        # Generate similarity assessments
        if 'party_info' in enriched:
            enriched['similarity_assessment'] = await self._generate_similarity_assessment(
                enriched['party_info']
            )

        # Generate commercial context
        if 'commercial_context' not in enriched:
            enriched['commercial_context'] = await self._generate_commercial_context(
                enriched
            )

        # Generate outcome prediction
        if 'outcome' not in enriched:
            enriched['outcome'] = await self._predict_outcome(enriched)

        return enriched

    async def _generate_similarity_assessment(
        self,
        party_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate mark and goods similarity assessments"""
        try:
            assessment = {
                'mark_similarity': {
                    'vis_sim': await self._calculate_visual_similarity(
                        party_info['app_mark'],
                        party_info['opp_mark']
                    ),
                    'aur_sim': await self._calculate_aural_similarity(
                        party_info['app_mark'],
                        party_info['opp_mark']
                    ),
                    'con_sim': await self._calculate_conceptual_similarity(
                        party_info['app_mark'],
                        party_info['opp_mark']
                    )
                },
                'gds_sim': await self._calculate_goods_similarity(party_info)
            }

            # Validate confidence thresholds
            self._validate_similarity_confidence(assessment)

            return assessment

        except Exception as e:
            raise MetadataError(
                message=f"Failed to generate similarity assessment: {str(e)}",
                error_code=ErrorCode.PREDICTIVE_FIELD_ERROR
            )

    def _validate_similarity_confidence(
        self,
        assessment: Dict[str, Any]
    ):
        """Validate similarity scores against confidence thresholds"""
        for field, threshold in self.confidence_thresholds.items():
            if field in assessment:
                score = self._calculate_confidence(assessment[field])
                if score < threshold:
                    raise MetadataError(
                        message=f"Confidence score below threshold for {field}",
                        error_code=ErrorCode.CONFIDENCE_THRESHOLD_ERROR,
                        field_path=field,
                        details={'score': score, 'threshold': threshold}
                    )

    async def _calculate_visual_similarity(
        self,
        mark1: str,
        mark2: str
    ) -> int:
        """Calculate visual similarity between marks"""
        # Implementation would use computer vision or string similarity
        # This is a placeholder returning a mock score
        return 3

    async def _calculate_aural_similarity(
        self,
        mark1: str,
        mark2: str
    ) -> int:
        """Calculate aural/phonetic similarity between marks"""
        # Implementation would use phonetic algorithms
        # This is a placeholder returning a mock score
        return 4

    async def _calculate_conceptual_similarity(
        self,
        mark1: str,
        mark2: str
    ) -> int:
        """Calculate conceptual similarity between marks"""
        # Implementation would use NLP techniques
        # This is a placeholder returning a mock score
        return 3

    def _get_cache_key(self, metadata: Dict[str, Any]) -> str:
        """Generate cache key for metadata validation"""
        # Create a stable representation for caching
        return json.dumps(metadata, sort_keys=True)

class VectorStoreManager:
    """
    Enhanced vector store manager with metadata support and validation.
    Handles vector operations with proper metadata handling and error recovery.
    """
    def __init__(
        self,
        api_key: str,
        environment: str,
        metadata_processor: MetadataProcessor,
        max_connections: int = 10,
        connection_timeout: int = 30,
        max_retries: int = 3
    ):
        """Initialize with metadata processing capabilities"""
        self.api_key = api_key
        self.environment = environment
        self.metadata_processor = metadata_processor
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        
        self._connection_pool = deque(maxlen=max_connections)
        self._pool_lock = asyncio.Lock()
        self._active_connections = {}
        self._last_cleanup = datetime.now()

    @asynccontextmanager
    async def get_connection(self) -> Index:
        """Get a connection from the pool with enhanced error handling"""
        async with self._pool_lock:
            await self._cleanup_expired_connections()
            
            while self._connection_pool:
                index, last_used = self._connection_pool.popleft()
                if (datetime.now() - last_used) < timedelta(minutes=5):
                    self._active_connections[id(index)] = datetime.now()
                    try:
                        yield index
                    finally:
                        self._connection_pool.append((index, datetime.now()))
                        del self._active_connections[id(index)]
                        return

            try:
                index = await self._create_connection()
                self._active_connections[id(index)] = datetime.now()
                yield index
            finally:
                self._connection_pool.append((index, datetime.now()))
                del self._active_connections[id(index)]

    async def _create_connection(self) -> Index:
        """Create a new connection with enhanced error handling"""
        try:
            pc = Pinecone(api_key=self.api_key)
            spec = ServerlessSpec(cloud="aws", region="eu-west-1")
            return pc.Index("default", spec)
        except Exception as e:
            raise StorageError(
                message=f"Failed to create Pinecone connection: {str(e)}",
                error_code=ErrorCode.CONNECTION_FAILED
            )

    async def upsert_with_metadata(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str,
        chunk_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upsert vectors with validated and enriched metadata.
        
        Args:
            vectors: List of vectors with metadata
            namespace: Target namespace
            chunk_id: Optional chunk identifier
            
        Returns:
            Operation results
            
        Raises:
            StorageError: If upsert fails
            SchemaValidationError: If metadata validation fails
        """
        processed_vectors = []
        
        for vector in vectors:
            # Process and validate metadata
            try:
                enriched_metadata = await self.metadata_processor.process_metadata(
                    vector['metadata'],
                    namespace,
                    chunk_id
                )
                
                processed_vectors.append({
                    'id': vector['id'],
                    'values': vector['values'],
                    'metadata': enriched_metadata
                })
                
            except (SchemaValidationError, MetadataError) as e:
                logger.error(f"Metadata processing failed for vector {vector['id']}: {str(e)}")
                if not self.metadata_processor.enable_validation:
                    # If validation is disabled, proceed with original metadata
                    processed_vectors.append(vector)
                continue

        if not processed_vectors:
            raise StorageError(
                message="No valid vectors to upsert",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                namespace=namespace
            )

        try:
            async with self.get_connection() as index:
                await index.upsert(
                    vectors=processed_vectors,
                    namespace=namespace
                )
                
            return {
                "status": "success",
                "vectors_processed": len(processed_vectors),
                "original_count": len(vectors)
            }

        except Exception as e:
            raise StorageError(
                message=f"Vector upsert failed: {str(e)}",
                error_code=ErrorCode.UPSERT_FAILED,
                namespace=namespace
            )

class QueryEnhancer:
    """
    Enhances vector queries with metadata filtering and validation.
    Provides advanced query capabilities for enriched metadata.
    """
    def __init__(
        self,
        metadata_processor: MetadataProcessor
    ):
        """Initialize with metadata processor"""
        self.metadata_processor = metadata_processor

    async def enhance_query(
        self,
        base_query: Dict[str, Any],
        metadata_filters: Optional[Dict[str, Any]] = None,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Enhance query with metadata filters and confidence thresholds.
        
        Args:
            base_query: Base vector query
            metadata_filters: Optional metadata filtering criteria
            confidence_threshold: Optional confidence score threshold
            
        Returns:
            Enhanced query
        """
        enhanced_query = base_query.copy()

        if metadata_filters:
            # Validate metadata filters against schema
            if self.metadata_processor.enable_validation:
                self._validate_filters(metadata_filters)

            # Add metadata filters to query
            enhanced_query['filter'] = self._build_filter_dict(metadata_filters)

        if confidence_threshold is not None:
            # Add confidence threshold to filters
            if 'filter' not in enhanced_query:
                enhanced_query['filter'] = {}
            enhanced_query['filter']['confidence_score'] = {
                '$gte': confidence_threshold
            }

        return enhanced_query

    def _validate_filters(self, filters: Dict[str, Any]):
        """Validate metadata filters against schema"""
        try:
            # Extract relevant schema parts for filters
            filter_schema = {
                'type': 'object',
                'properties': {
                    k: self.metadata_processor.schema['properties'][k]
                    for k in filters.keys()
                    if k in self.metadata_processor.schema['properties']
                }
            }
            
            validate(instance=filters, schema=filter_schema)
            
        except ValidationError as e:
            raise SchemaValidationError(
                message="Invalid metadata filters",
                field_path=str(e.path),
                expected_type=str(e.validator),
                received_value=e.instance
            )

    def _build_filter_dict(
        self,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build Pinecone filter dictionary from metadata filters"""
        filter_dict = {}
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle range queries
                if 'gte' in value or 'lte' in value:
                    filter_dict[key] = {}
                    if 'gte' in value:
                        filter_dict[key]['$gte'] = value['gte']
                    if 'lte' in value:
                        filter_dict[key]['$lte'] = value['lte']
            else:
                # Handle exact match queries
                filter_dict[key] = {'$eq': value}
                
        return filter_dict

class BatchProcessor:
    """
    Handles batch processing of vectors and metadata with validation.
    Provides efficient batch operations with error recovery and statistics tracking.
    """
    def __init__(
        self,
        vector_store: VectorStoreManager,
        metadata_processor: MetadataProcessor,
        batch_size: int = 100,
        max_retries: int = 3
    ):
        """Initialize with required components"""
        self.vector_store = vector_store
        self.metadata_processor = metadata_processor
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.stats = {
            'processed': 0,
            'failed': 0,
            'retried': 0,
            'validation_errors': 0
        }

    async def process_batch(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str
    ) -> Dict[str, Any]:
        """
        Process a batch of vectors with metadata validation and error recovery.
        
        Args:
            vectors: List of vectors with metadata
            namespace: Target namespace
            
        Returns:
            Batch processing results
        """
        batches = self._create_batches(vectors)
        results = []
        
        for batch in batches:
            try:
                batch_result = await self._process_single_batch(
                    batch,
                    namespace
                )
                results.append(batch_result)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {str(e)}")
                self.stats['failed'] += 1
                
                # Attempt to process individual vectors
                await self._handle_batch_failure(batch, namespace)

        return self._summarize_results(results)

    def _create_batches(
        self,
        vectors: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Split vectors into batches"""
        return [
            vectors[i:i + self.batch_size]
            for i in range(0, len(vectors), self.batch_size)
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _process_single_batch(
        self,
        batch: List[Dict[str, Any]],
        namespace: str
    ) -> Dict[str, Any]:
        """Process a single batch with retry logic"""
        try:
            result = await self.vector_store.upsert_with_metadata(
                batch,
                namespace
            )
            
            self.stats['processed'] += len(batch)
            return result
            
        except Exception as e:
            self.stats['retried'] += 1
            raise

    async def _handle_batch_failure(
        self,
        batch: List[Dict[str, Any]],
        namespace: str
    ):
        """Handle batch failure by processing vectors individually"""
        for vector in batch:
            try:
                await self.vector_store.upsert_with_metadata(
                    [vector],
                    namespace
                )
                self.stats['processed'] += 1
                
            except SchemaValidationError:
                self.stats['validation_errors'] += 1
            except Exception:
                self.stats['failed'] += 1

    def _summarize_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarize batch processing results"""
        return {
            'total_processed': self.stats['processed'],
            'total_failed': self.stats['failed'],
            'total_retried': self.stats['retried'],
            'validation_errors': self.stats['validation_errors'],
            'success_rate': (
                self.stats['processed'] /
                (self.stats['processed'] + self.stats['failed'])
                if (self.stats['processed'] + self.stats['failed']) > 0
                else 0
            )
        }

async def cleanup_namespace(
    vector_store: VectorStoreManager,
    namespace: str
) -> Dict[str, Any]:
    """
    Clean up a namespace with proper error handling.
    
    Args:
        vector_store: VectorStoreManager instance
        namespace: Namespace to clean up
        
    Returns:
        Cleanup results
    """
    try:
        async with vector_store.get_connection() as index:
            await index.delete(
                deleteAll=True,
                namespace=namespace
            )
            
        return {
            'status': 'success',
            'namespace': namespace,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise StorageError(
            message=f"Failed to clean up namespace: {str(e)}",
            error_code=ErrorCode.NAMESPACE_ERROR,
            namespace=namespace
        )

# Initialize global components
_vector_store_manager: Optional[VectorStoreManager] = None
_metadata_processor: Optional[MetadataProcessor] = None
_query_enhancer: Optional[QueryEnhancer] = None

def init_components(
    config: Dict[str, Any]
) -> Tuple[VectorStoreManager, MetadataProcessor, QueryEnhancer]:
    """
    Initialize global components with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of initialized components
    """
    global _vector_store_manager, _metadata_processor, _query_enhancer
    
    if _metadata_processor is None:
        _metadata_processor = MetadataProcessor(
            schema=config['SCHEMA'],
            confidence_thresholds=config['CONFIDENCE_THRESHOLDS'],
            enable_validation=config['ENABLE_VALIDATION']
        )
    
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager(
            api_key=config['PINECONE_API_KEY'],
            environment=config['PINECONE_ENVIRONMENT'],
            metadata_processor=_metadata_processor
        )
    
    if _query_enhancer is None:
        _query_enhancer = QueryEnhancer(
            metadata_processor=_metadata_processor
        )
    
    return _vector_store_manager, _metadata_processor, _query_enhancer

# Convenience functions for common operations
async def upsert_vectors(
    vectors: List[Dict[str, Any]],
    namespace: str,
    chunk_id: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for upserting vectors with metadata"""
    if _vector_store_manager is None:
        raise RuntimeError("Components not initialized. Call init_components first.")
        
    return await _vector_store_manager.upsert_with_metadata(
        vectors,
        namespace,
        chunk_id
    )

async def process_batch(
    vectors: List[Dict[str, Any]],
    namespace: str,
    batch_size: int = 100
) -> Dict[str, Any]:
    """Convenience function for batch processing"""
    if _vector_store_manager is None or _metadata_processor is None:
        raise RuntimeError("Components not initialized. Call init_components first.")
        
    processor = BatchProcessor(
        vector_store=_vector_store_manager,
        metadata_processor=_metadata_processor,
        batch_size=batch_size
    )
    
    return await processor.process_batch(vectors, namespace)

async def enhance_query(
    query: Dict[str, Any],
    metadata_filters: Optional[Dict[str, Any]] = None,
    confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Convenience function for query enhancement"""
    if _query_enhancer is None:
        raise RuntimeError("Components not initialized. Call init_components first.")
        
    return await _query_enhancer.enhance_query(
        query,
        metadata_filters,
        confidence_threshold
    )