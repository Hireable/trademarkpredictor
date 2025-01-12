"""
Shared test fixtures for the trademark predictor application.
"""

import json
from pathlib import Path
from typing import AsyncGenerator, Dict, Any
import pytest
from unittest.mock import Mock, AsyncMock

from src.config.settings import Settings
from src.agent.trademark import TrademarkAgent
from src.storage.pinecone import PineconeStorage
from src.agent.processors.document import DocumentProcessor
from src.agent.processors.json import JsonProcessor
from src.config.base import ProcessingConfig

@pytest.fixture
def sample_pdf_path() -> Path:
    """Fixture providing path to sample PDF."""
    return Path("data/raw_pdfs/JOLLY PECKISH.pdf")

@pytest.fixture
def mock_settings() -> Mock:
    """Fixture providing mocked settings."""
    settings = Mock(spec=Settings)
    settings.pinecone = Mock()
    settings.pinecone.api_key = "test_api_key"
    settings.pinecone.environment = "test-env"
    settings.pinecone.chunks_index = "test-index"
    settings.pinecone.location = "test-location"
    settings.pinecone.batch_size = 100
    
    settings.vertex_ai = Mock()
    settings.vertex_ai.project_id = "test-project"
    settings.vertex_ai.location = "test-location"
    settings.vertex_ai.model_name = "gemini-1.5-pro"
    settings.vertex_ai.max_tokens = 4000
    settings.vertex_ai.temperature = 0.2
    
    settings.schema_path = Path("data/schema.json")
    settings.examples_path = Path("data/examples.json")
    settings.processing = Mock()
    settings.processing.chunk_size = 1500
    settings.processing.chunk_overlap = 150
    
    return settings

@pytest.fixture
def mock_storage() -> Mock:
    """Fixture providing mocked Pinecone storage."""
    storage = Mock(spec=PineconeStorage)
    storage.initialize = AsyncMock()
    storage.upsert = AsyncMock()
    storage.query = AsyncMock()
    storage.delete = AsyncMock()
    storage.cleanup = AsyncMock()
    return storage

@pytest.fixture
def mock_document_processor() -> Mock:
    """Fixture providing mocked document processor."""
    processor = Mock(spec=DocumentProcessor)
    processor.process_pdf = AsyncMock()
    return processor

@pytest.fixture
def mock_json_processor() -> Mock:
    """Fixture providing mocked JSON processor."""
    processor = Mock(spec=JsonProcessor)
    processor.generate_json = AsyncMock()
    processor.validator = Mock()
    return processor

@pytest.fixture
def processing_config() -> ProcessingConfig:
    """Fixture providing processing configuration."""
    return ProcessingConfig(
        chunk_size=1500,
        chunk_overlap=150,
        max_retries=3,
        batch_size=5
    )

@pytest.fixture
async def trademark_agent(
    mock_settings: Mock,
    mock_storage: Mock,
    mock_document_processor: Mock,
    mock_json_processor: Mock
) -> AsyncGenerator[TrademarkAgent, None]:
    """
    Fixture providing configured TrademarkAgent with mocked dependencies.
    
    Args:
        mock_settings: Mocked settings
        mock_storage: Mocked storage
        mock_document_processor: Mocked document processor
        mock_json_processor: Mocked JSON processor
        
    Yields:
        Configured TrademarkAgent instance
    """
    agent = TrademarkAgent()
    agent.storage = mock_storage
    agent.document_processor = mock_document_processor
    agent.json_processor = mock_json_processor
    
    # Mock resources
    agent.resources = Mock()
    agent.resources.embedding_model = Mock()
    agent.resources.embedding_model.encode = Mock(return_value=[0.1] * 1024)
    agent.resources.gemini_model = Mock()
    
    yield agent
    await agent.cleanup()

@pytest.fixture
def sample_trademark_data() -> Dict[str, Any]:
    """Fixture providing sample trademark decision data."""
    return {
        "case_metadata": {
            "case_ref": "O/0703/24",
            "officer": "Sarah Wallace",
            "dec_date": "2024-07-25",
            "jurisdiction": "UK"
        },
        "party_info": {
            "app_mark": "JOLLY PECKISH",
            "opp_mark": "JOLLY",
            "app_name": "Stonegate Farmers Limited",
            "opp_name": "The Jolly Hog Group Limited",
            "app_wipo_basic_numbers": ["3828698"],
            "opp_wipo_basic_numbers": ["918282195"]
        },
        "commercial_context": {
            "app_spec": "Eggs, Birds egg products, Dairy products",
            "opp_spec": "Meat, fish, poultry and game",
            "app_class": [29, 30],
            "opp_class": [29, 30, 43]
        }
    }