CONFIG = {
    "PROJECT_ID": "trademark-case-agent",
    "INDEX_NAME": "trademark-embeddings-index",
    "DIMENSION": 768,
    "CLOUD": "aws",
    "REGION": "us-east-1",
    "SCHEMA_PATH": "data/schema.json",
    "ERROR_MESSAGES": {
        "SCHEMA_READ_ERROR": "Error reading schema file: {}",
        "EMBEDDING_FAILED": "Failed to generate embedding.",
        "UNEXPECTED_ERROR": "Unexpected error occurred: {}",
    },
}


def get_config():
    """
    Retrieves the configuration.
    Returns:
        dict: A dictionary containing configuration values.
    """
    return CONFIG.copy()