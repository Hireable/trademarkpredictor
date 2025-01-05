import os
from typing import Dict, Any

# Configuration dictionary containing all settings for the trademark vector system
CONFIG: Dict[str, Any] = {
    # Project identification
    "PROJECT_ID": "trademark-case-agent",
    
    # Pinecone configuration
    "INDEX_NAME": "trademark-embeddings-index",
    "CLOUD": "aws",
    "REGION": "us-east-1",
    "DIMENSION": 768,  # Dimension size for the embedding vector
    
    # File paths
    "SCHEMA_PATH": r"C:\Users\jpbpr\OneDrive\Documents\Hireable\LegalTech Predictor\tm-case-agent-repo\tmpredictorschema.json",
    
    # Embedding configuration
    "EMBEDDING_MODEL": "text-embedding-3-large",
    
    # Error messages
    "ERROR_MESSAGES": {
        "EMBEDDING_FAILED": "Embedding generation failed. No embedding returned.",
        "SCHEMA_READ_ERROR": "Error reading schema file: {}",
        "VALIDATION_ERROR": "Case data validation failed: {}",
        "NETWORK_ERROR": "Network error during case processing: {}",
        "UNEXPECTED_ERROR": "Unexpected error during case processing: {}"
    }
}

def get_config() -> Dict[str, Any]:
    """
    Retrieves the configuration with any environment-specific modifications.
    
    Returns:
        Dict[str, Any]: The complete configuration dictionary
    """
    # Create a copy of the config to avoid modifying the original
    current_config = CONFIG.copy()
    
    # Override with environment variables if they exist
    if "PINECONE_API_KEY" in os.environ:
        current_config["PINECONE_API_KEY"] = os.environ["PINECONE_API_KEY"]
    
    return current_config