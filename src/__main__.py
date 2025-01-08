import os
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from src.agent import TrademarkCaseAgent, ProcessingConfig
from src.config import get_config, VertexAIConfig
from src.exceptions import ConfigurationError

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

async def verify_environment() -> None:
    """
    Verify and log environment configuration status.
    This helps with debugging configuration issues before running the main process.
    """
    logger.info("Verifying environment configuration...")

    # Check essential environment variables
    api_key = os.getenv('PINECONE_API_KEY')
    env = os.getenv('PINECONE_ENVIRONMENT')
    project_id = os.getenv('VERTEX_PROJECT_ID')
    credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    logger.info("Environment Variable Status:")
    logger.info(f"PINECONE_API_KEY exists: {bool(api_key)}")
    logger.info(f"PINECONE_API_KEY length: {len(api_key) if api_key else 0}")
    logger.info(f"PINECONE_ENVIRONMENT exists: {bool(env)}")
    logger.info(f"PINECONE_ENVIRONMENT value: {env if env else 'Not found'}")
    logger.info(f"VERTEX_PROJECT_ID exists: {bool(project_id)}")
    logger.info(f"VERTEX_PROJECT_ID value: {project_id if project_id else 'Not found'}")
    logger.info(f"GOOGLE_APPLICATION_CREDENTIALS exists: {bool(credentials)}")
    logger.info(f"GOOGLE_APPLICATION_CREDENTIALS value: {credentials if credentials else 'Not found'}")

    # Verify configuration
    try:
        config = get_config()
        logger.info("\nConfiguration Status:")
        for key, value in config.items():
            if 'API_KEY' in key or 'PROJECT_ID' in key:
                logger.info(f"{key}: {'[EXISTS]' if value else '[MISSING]'}")
            else:
                logger.info(f"{key}: {value}")
    except Exception as e:
        raise ConfigurationError(f"Configuration verification failed: {str(e)}")

async def process_directory(
    agent: TrademarkCaseAgent,
    directory: Path,
    batch_size: Optional[int] = None
) -> None:
    """
    Process all PDF files in the specified directory concurrently.

    Args:
        agent: Initialized TrademarkCaseAgent
        directory: Path to directory containing PDF files
        batch_size: Optional limit on concurrent processing
    """
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process files concurrently in batches
    for i in range(0, len(pdf_files), batch_size or 5):
        batch = pdf_files[i:i + (batch_size or 5)]
        tasks = [
            agent.process_pdf_streaming(str(pdf_file))
            for pdf_file in batch
        ]
        await asyncio.gather(*tasks)
        logger.info(f"Completed batch {i // (batch_size or 5) + 1}")

async def main():
    """Main async function to run the trademark processing pipeline"""
    parser = argparse.ArgumentParser(description="Process trademark case documents")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=os.path.join("data", "raw_pdfs"),
        help="Directory containing PDF files to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of PDFs to process concurrently"
    )
    args = parser.parse_args()

    try:
        # Verify environment and configuration
        await verify_environment()

        # Get configuration
        config = get_config()

        # Create processing configuration
        processing_config = ProcessingConfig(
            batch_size=args.batch_size,
            chunk_size=config["CHUNK_SIZE"],
            chunk_overlap=config["CHUNK_OVERLAP"],
            max_retries=config["MAX_RETRIES"],
            wipo_confidence_threshold=config["WIPO_CONFIDENCE_THRESHOLD"],
            embedding_dimension=config["EMBEDDING_DIMENSION"]
        )
        
        # Create Vertex AI configuration
        vertex_ai_config = VertexAIConfig.from_env()

        # Initialize agent with custom configuration
        agent = TrademarkCaseAgent(
            config=config,
            processing_config=processing_config,
            vertex_ai_config=vertex_ai_config
        )

        try:
            # Process PDF directory
            pdf_dir = Path(args.pdf_dir)
            if not pdf_dir.exists():
                raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

            await process_directory(
                agent,
                pdf_dir,
                batch_size=args.batch_size
            )

            logger.info("Processing pipeline completed successfully")

        finally:
            # Ensure proper cleanup of resources
            await agent.cleanup()

    except Exception as e:
        logger.error(f"Processing pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())