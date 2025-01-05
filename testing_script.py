# tests/conftest.py
import pytest
from unittest.mock import Mock, patch
import json
from typing import Dict, Any

@pytest.fixture
def mock_config():
    """Provides a test configuration that doesn't depend on external files"""
    return {
        "PROJECT_ID": "test-project",
        "INDEX_NAME": "test-index",
        "CLOUD": "aws",
        "REGION": "us-east-1",
        "DIMENSION": 768,
        "SCHEMA_PATH": "test_schema.json",
        "ERROR_MESSAGES": {
            "EMBEDDING_FAILED": "Embedding generation failed",
            "SCHEMA_READ_ERROR": "Error reading schema: {}",
            "VALIDATION_ERROR": "Validation failed: {}",
            "NETWORK_ERROR": "Network error: {}",
            "UNEXPECTED_ERROR": "Unexpected error: {}"
        }
    }

@pytest.fixture
def sample_case_data():
    """Provides sample case data for testing"""
    return {
        "case_metadata": {
            "case_ref": "TEST001",
            "officer": "Test Officer",
            "dec_date": "2024-01-05",
            "jurisdiction": "TEST"
        },
        "party_info": {
            "app_mark": "TestBrand",
            "opp_mark": "TestMark",
            "app_name": "Test Applicant",
            "opp_name": "Test Opponent",
            "market_presence": {
                "opp_market_tenure": 3,
                "geographic_overlap": 2
            }
        },
        "commercial_context": {
            "app_spec": "Test Goods",
            "opp_spec": "Test Services",
            "app_class": [42],
            "opp_class": [42],
            "market_characteristics": {
                "price_point": 2,
                "purchase_frequency": 3,
                "market_sophistication": 2
            }
        },
        "similarity_assessment": {
            "mark_similarity": {
                "vis_sim": 3,
                "aur_sim": 2,
                "con_sim": 4
            },
            "gds_sim": {
                "nature": 3,
                "purpose": 4,
                "channels": 2,
                "use": 3
            }
        },
        "outcome": {
            "confusion": False,
            "conf_type": "none",
            "confidence_score": 3
        }
    }
