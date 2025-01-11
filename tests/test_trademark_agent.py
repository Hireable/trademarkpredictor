import pytest
import asyncio
import os
from typing import Dict, Any, List
import logging
from datetime import datetime
from pathlib import Path
import json

from pinecone import Pinecone
import numpy as np
from dotenv import load_dotenv

from ..src.agent import TrademarkCaseAgent
from ..src.config import ProcessingConfig
from ..src.utils import MetadataProcessor, VectorStoreManager
from ..src.exceptions import StorageError

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeTestValidator:
    """Helper class to validate Pinecone operations and data structure"""
    
    def __init__(self, api_key: str, environment: str):
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.environment = environment
        
    async def validate_index_structure(self, index_name: str) -> Dict[str, Any]:
        """Validates the index configuration and returns its statistics"""
        try:
            index = self.pc.Index(index_name)
            stats = await index.describe_index_stats()
            logger.info(f"Index statistics: {json.dumps(stats, indent=2)}")
            return stats
        except Exception as e:
            logger.error(f"Failed to validate index structure: {str(e)}")
            raise

@pytest.mark.asyncio
class TestPineconeIntegration:
    """Integration tests for Pinecone operations"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Set up test environment and resources"""
        # Load environment variables
        load_dotenv()
        
        # Load test configuration
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.environment = os.getenv('PINECONE_ENVIRONMENT')
        
        if not self.api_key or not self.environment:
            pytest.skip("Pinecone credentials not configured")
        
        # Load test schema and examples
        test_data_dir = Path(__file__).parent / "data"
        self.schema_path = test_data_dir / "test_schema.json"
        self.examples_path = test_data_dir / "test_examples.json"
        
        # Ensure test data directory exists
        test_data_dir.mkdir(exist_ok=True)
        
        # Create test schema if it doesn't exist
        if not self.schema_path.exists():
            test_schema = {
                "type": "object",
                "properties": {
                    "case_metadata": {"type": "object"},
                    "party_info": {"type": "object"},
                    "commercial_context": {"type": "object"}
                },
                "required": ["case_metadata", "party_info", "commercial_context"]
            }
            with open(self.schema_path, 'w') as f:
                json.dump(test_schema, f, indent=2)
        
        # Load schema
        with open(self.schema_path) as f:
            self.test_schema = json.load(f)
        
        # Initialize validator
        self.validator = PineconeTestValidator(
            self.api_key,
            self.environment
        )
        
        # Create test namespace
        self.test_namespace = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        yield
        
        # Cleanup test namespace
        try:
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc.Index("tm-doc-chunks")
            await index.delete(
                deleteAll=True,
                namespace=self.test_namespace
            )
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    async def test_index_structure(self):
        """Verify index configuration and structure"""
        stats = await self.validator.validate_index_structure("tm-doc-chunks")
        
        # Verify index properties
        assert stats['dimension'] == 1536, "Incorrect embedding dimension"
        assert stats['total_vector_count'] > 0, "Index is empty"
        
        logger.info(f"Index statistics: {json.dumps(stats, indent=2)}")
    
    async def test_document_processing_and_storage(self):
        """Test complete document processing pipeline"""
        # Initialize agent with test configuration
        config = {
            "SCHEMA": self.test_schema,
            "CONFIDENCE_THRESHOLDS": {"mark_similarity": 0.7, "gds_sim": 0.75},
            "ENABLE_VALIDATION": True
        }
        
        agent = TrademarkCaseAgent(config=config)
        
        try:
            # Find test PDF
            test_pdfs_dir = Path(__file__).parent / "test_pdfs"
            test_pdf = next(test_pdfs_dir.glob("*.pdf"), None)
            
            if not test_pdf:
                pytest.skip("Test document not found")
            
            # Process document
            await agent.process_pdf_streaming(
                str(test_pdf),
                namespace=self.test_namespace
            )
            
            # Allow time for processing
            await asyncio.sleep(5)
            
            # Verify results
            metadata_stats = await self.validator.verify_vector_metadata(
                "tm-doc-chunks",
                self.test_namespace,
                ["case_metadata", "party_info", "commercial_context"]
            )
            
            assert metadata_stats['compliant_vectors'] > 0, "No compliant vectors found"
            
        finally:
            await agent.cleanup()
    
    async def test_batch_upsert_validation(self):
        """Test batch upsert operations"""
        manager = VectorStoreManager(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=self.environment,
            metadata_processor=MetadataProcessor(
                schema=self.test_schema,
                confidence_thresholds={"mark_similarity": 0.7, "gds_sim": 0.75}
            )
        )
        
        # Create test vectors
        test_vectors = [
            {
                'id': f'test_vector_{i}',
                'values': np.random.rand(1536).tolist(),
                'metadata': {
                    'case_metadata': {'reference': f'TM{i:03d}'},
                    'party_info': {'app_mark': f'TestMark{i}'},
                    'commercial_context': {'goods_services': ['Class 9']}
                }
            }
            for i in range(10)
        ]
        
        try:
            # Perform batch upsert
            result = await manager.upsert_with_metadata(
                test_vectors,
                self.test_namespace
            )
            
            assert result['status'] == 'success'
            
        finally:
            await manager.cleanup()
    
    async def test_query_validation(self):
        """Test query operations"""
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index("tm-doc-chunks")
        
        try:
            response = await index.query(
                namespace=self.test_namespace,
                vector=np.random.rand(1536).tolist(),
                top_k=5,
                include_metadata=True
            )
            
            assert hasattr(response, 'matches'), "Invalid response structure"
            
            for match in response.matches:
                assert hasattr(match, 'id'), "Match missing ID"
                assert hasattr(match, 'metadata'), "Match missing metadata"
                
                metadata = match.metadata
                assert 'case_metadata' in metadata
                assert 'party_info' in metadata
                
        except Exception as e:
            logger.error(f"Query validation failed: {str(e)}")
            raise