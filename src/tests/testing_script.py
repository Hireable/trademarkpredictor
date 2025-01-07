# tests/test_trademark_agent.py
import pytest
from unittest.mock import Mock, patch
import torch
from src.test_agent import TrademarkCaseAgent

@pytest.fixture
def mock_pinecone():
    """Mock Pinecone client"""
    with patch('pinecone.Pinecone') as mock_pc:
        # Configure the mock to return our test index name
        mock_pc.return_value.list_indexes.return_value.names.return_value = []
        mock_pc.return_value.Index.return_value.upsert = Mock(return_value=True)
        yield mock_pc

@pytest.fixture
def mock_storage():
    """Mock Google Cloud Storage client"""
    with patch('google.cloud.storage.Client') as mock_storage:
        yield mock_storage

@pytest.fixture
def mock_documentai():
    """Mock Document AI client"""
    with patch('google.cloud.documentai_v1.DocumentProcessorServiceClient') as mock_doc:
        yield mock_doc

@pytest.fixture
def agent(mock_config, mock_pinecone, mock_storage, mock_documentai):
    """Create a TrademarkCaseAgent instance with mocked dependencies"""
    with patch('test_agent.get_config', return_value=mock_config):
        with patch('torch.no_grad'):
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                with patch('transformers.AutoModel.from_pretrained') as mock_model:
                    # Configure the mock model to return a valid embedding
                    mock_output = Mock()
                    mock_output.last_hidden_state = torch.zeros((1, 1, 768))
                    mock_model.return_value.return_value = mock_output
                    
                    agent = TrademarkCaseAgent()
                    return agent

class TestTrademarkCaseAgent:
    def test_validate_case_data(self, agent, sample_case_data):
        """Test case data validation"""
        # Test with valid data
        assert agent.validate_case_data(sample_case_data) is True
        
        # Test with missing required keys
        invalid_data = sample_case_data.copy()
        del invalid_data['case_metadata']
        assert agent.validate_case_data(invalid_data) is False

    def test_flatten_metadata(self, agent, sample_case_data):
        """Test metadata flattening"""
        flat_data = agent._flatten_metadata(sample_case_data)
        
        # Check that nested structures are flattened
        assert 'party_info_market_presence_opp_market_tenure' in flat_data
        assert isinstance(flat_data['party_info_market_presence_opp_market_tenure'], int)
        
        # Check that arrays of primitives are preserved
        assert isinstance(flat_data['commercial_context_app_class'], list)

    def test_generate_embedding(self, agent):
        """Test embedding generation"""
        test_text = "Sample trademark case text"
        embedding = agent.generate_embedding(test_text)
        
        assert embedding is not None
        assert len(embedding) == 768  # LEGAL-BERT dimension
        assert isinstance(embedding, list)

    def test_create_text_representation(self, agent, sample_case_data):
        """Test text representation creation"""
        text = agent._create_text_representation(sample_case_data)
        
        # Check that all important elements are included
        assert sample_case_data['case_metadata']['case_ref'] in text
        assert sample_case_data['party_info']['app_mark'] in text
        assert sample_case_data['party_info']['opp_mark'] in text
        assert str(sample_case_data['commercial_context']['app_class'][0]) in text

    def test_process_case(self, agent, sample_case_data):
        """Test end-to-end case processing"""
        result = agent.process_case(sample_case_data)
        
        assert result['success'] is True
        assert result['case_ref'] == sample_case_data['case_metadata']['case_ref']

    def test_process_case_invalid_data(self, agent):
        """Test case processing with invalid data"""
        invalid_data = {'incomplete': 'data'}
        result = agent.process_case(invalid_data)
        
        assert result['success'] is False
        assert 'error' in result

    def test_store_in_pinecone(self, agent, sample_case_data):
        """Test Pinecone storage"""
        embedding = [0.0] * 768  # Mock embedding
        success = agent.store_in_pinecone(
            case_id='TEST001',
            embedding=embedding,
            metadata=sample_case_data
        )
        
        assert success is True

if __name__ == '__main__':
    pytest.main(['-v'])