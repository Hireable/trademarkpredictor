import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Index names
PINECONE_INDEX_NAME = "tm-case-chunks" # For storing PDF chunks
WIPO_INDEX_NAME = "wipo-mapping"          # For WIPO mapping

# Embedding model configuration
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
EMBEDDING_DIMENSION = 768

# Langchain text splitter configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Other settings
SCHEMA_PATH = os.path.join("data", "schema.json")
WIPO_MAPPING_FILE = os.path.join("data", "wipo_mapping.xlsx")

def get_config() -> Dict[str, Any]:
    """Returns the application configuration as a dictionary."""
    return {
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "PINECONE_ENVIRONMENT": PINECONE_ENVIRONMENT,
        "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
        "WIPO_INDEX_NAME": WIPO_INDEX_NAME,
        "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
        "EMBEDDING_DIMENSION": EMBEDDING_DIMENSION,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "SCHEMA_PATH": SCHEMA_PATH,
        "WIPO_MAPPING_FILE": WIPO_MAPPING_FILE,
    }