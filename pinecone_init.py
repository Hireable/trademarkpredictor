import asyncio
import logging
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def validate_schema(schema_path: str) -> Dict[str, Any]:
    """Load and validate the schema configuration"""
    try:
        with open(schema_path) as f:
            schema = json.load(f)
            logger.info("Schema loaded successfully")
            return schema
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        raise

async def initialize_pinecone_indexes() -> None:
    """Initialize and validate Pinecone indexes"""
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        logger.info("Connected to Pinecone successfully")

        # Get index configurations from environment
        case_index_name = os.getenv('PINECONE_CHUNKS_INDEX')
        predictive_index_name = os.getenv('PINECONE_PREDICTIVE_INDEX')
        case_dimension = int(os.getenv('PINECONE_CHUNKS_DIMENSION'))
        predictive_dimension = int(os.getenv('PINECONE_PREDICTIVE_DIMENSION'))

        # Define serverless specification
        spec = ServerlessSpec(
            cloud="gcp",
            region="europe-west4"
        )

        # List existing indexes
        existing_indexes = pc.list_indexes()
        logger.info(f"Found existing indexes: {existing_indexes}")

        # Set up case chunks index
        if case_index_name not in existing_indexes:
            logger.info(f"Creating case index: {case_index_name}")
            pc.create_index(
                name=case_index_name,
                dimension=case_dimension,
                metric='cosine',
                spec=spec
            )
        else:
            logger.info(f"Validating existing case index: {case_index_name}")
            index_desc = pc.describe_index(case_index_name)
            if index_desc.dimension != case_dimension:
                raise ValueError(
                    f"Case index dimension mismatch: {index_desc.dimension} "
                    f"(expected {case_dimension})"
                )

        # Set up predictive index
        if predictive_index_name not in existing_indexes:
            logger.info(f"Creating predictive index: {predictive_index_name}")
            pc.create_index(
                name=predictive_index_name,
                dimension=predictive_dimension,
                metric='cosine',
                spec=spec
            )
        else:
            logger.info(f"Validating existing predictive index: {predictive_index_name}")
            index_desc = pc.describe_index(predictive_index_name)
            if index_desc.dimension != predictive_dimension:
                raise ValueError(
                    f"Predictive index dimension mismatch: {index_desc.dimension} "
                    f"(expected {predictive_dimension})"
                )

        # Test index connections
        case_index = pc.Index(case_index_name)
        pred_index = pc.Index(predictive_index_name)

        # Get index statistics
        case_stats = case_index.describe_index_stats()
        pred_stats = pred_index.describe_index_stats()

        logger.info(f"""
        Index Statistics:
        Case Index ({case_index_name}):
            Total vectors: {case_stats.total_vector_count}
            Namespaces: {len(case_stats.namespaces)}
            
        Predictive Index ({predictive_index_name}):
            Total vectors: {pred_stats.total_vector_count}
            Namespaces: {len(pred_stats.namespaces)}
        """)

        logger.info("Pinecone initialization completed successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise

async def main():
    """Main initialization function"""
    # Load environment variables
    load_dotenv()
    
    # Validate required environment variables
    required_vars = [
        'PINECONE_API_KEY',
        'PINECONE_ENVIRONMENT',
        'PINECONE_CHUNKS_INDEX',
        'PINECONE_PREDICTIVE_INDEX',
        'PINECONE_CHUNKS_DIMENSION',
        'PINECONE_PREDICTIVE_DIMENSION'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    # Initialize Pinecone
    await initialize_pinecone_indexes()

    # Validate schema
    schema_path = os.getenv('SCHEMA_PATH', 'data/schema.json')
    await validate_schema(schema_path)

if __name__ == "__main__":
    asyncio.run(main())