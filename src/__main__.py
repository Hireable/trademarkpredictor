import os
import sys
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv

from .agent import TrademarkCaseAgent
from .config import get_config, VertexAIConfig, ProcessingConfig
from .exceptions import ConfigurationError, ProcessingError

load_dotenv()

# Configure logging with enhanced format including process ID
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"trademark_processor_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnvironmentValidationResult:
    """Structured result of environment validation"""
    is_valid: bool
    messages: List[str]
    warnings: List[str]
    config: Optional[dict] = None

async def verify_environment() -> EnvironmentValidationResult:
    """
    Comprehensive environment verification that checks and validates all required
    configuration elements.
    
    The function performs the following validations:
    1. Checks for required environment variables
    2. Validates the format and content of each variable
    3. Verifies file paths and directory structures
    4. Tests connectivity to external services
    5. Validates configuration parameters
    
    Returns:
        EnvironmentValidationResult containing validation status and details
        
    Raises:
        ConfigurationError: If critical configuration elements are missing or invalid
    """
    messages = []
    warnings = []
    
    # Required environment variables with validation rules
    required_vars = {
        'PINECONE_API_KEY': {
            'min_length': 32,
            'validate': lambda x: len(x) >= 32,
            'error': 'PINECONE_API_KEY must be at least 32 characters long'
        },
        'PINECONE_ENVIRONMENT': {
            'validate': lambda x: x in ['gcp-starter', 'aws', 'azure'],
            'error': 'PINECONE_ENVIRONMENT must be one of: gcp-starter, aws, azure'
        },
        'VERTEX_PROJECT_ID': {
            'validate': lambda x: x.replace("-", "").isalnum(),
            'error': 'VERTEX_PROJECT_ID must contain only letters, numbers, and hyphens'
        },
        'GOOGLE_APPLICATION_CREDENTIALS': {
            'validate': lambda x: Path(x).exists() and Path(x).suffix == '.json',
            'error': 'GOOGLE_APPLICATION_CREDENTIALS must point to an existing JSON file'
        }
    }

    # Check and validate each required variable
    for var_name, rules in required_vars.items():
        value = os.getenv(var_name)
        
        if not value:
            messages.append(f"Missing required environment variable: {var_name}")
            continue
            
        # Perform validation if the variable exists
        if not rules['validate'](value):
            messages.append(rules['error'])
            continue
            
        # Log success without exposing sensitive values
        if 'API_KEY' in var_name or 'CREDENTIALS' in var_name:
            logging.info(f"{var_name}: [VALID]")
        else:
            logging.info(f"{var_name}: {value}")

    # Verify and create directory structure
    required_dirs = {
        'DATA_DIR': Path(os.getenv('DATA_DIR', 'data')),
        'PDF_DIR': Path(os.getenv('PDF_DIR', 'data/raw_pdfs')),
        'OUTPUT_DIR': Path(os.getenv('OUTPUT_DIR', 'data/processed')),
        'LOG_DIR': Path(os.getenv('LOG_DIR', 'logs'))
    }

    for dir_name, dir_path in required_dirs.items():
        if not dir_path.exists():
            warnings.append(f"{dir_name} not found: {dir_path}. Attempting to create.")
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                messages.append(f"Created {dir_name}: {dir_path}")
            except Exception as e:
                messages.append(f"Failed to create {dir_name}: {str(e)}")

    # Verify configuration
    try:
        config = get_config()
        
        # Validate embedding dimensions
        if config['CHUNK_EMBEDDING_DIMENSION'] <= 0:
            messages.append("Invalid CHUNK_EMBEDDING_DIMENSION in configuration")
        if config['METADATA_EMBEDDING_DIMENSION'] <= 0:
            messages.append("Invalid METADATA_EMBEDDING_DIMENSION in configuration")
            
        # Validate processing parameters
        if config['CHUNK_SIZE'] <= config['CHUNK_OVERLAP']:
            messages.append("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
        if config['BATCH_SIZE'] <= 0:
            messages.append("BATCH_SIZE must be positive")
            
    except Exception as e:
        messages.append(f"Configuration verification failed: {str(e)}")
        config = None

    # Test external service connectivity
    await _test_external_services(messages)

    is_valid = len(messages) == 0
    return EnvironmentValidationResult(
        is_valid=is_valid,
        messages=messages,
        warnings=warnings,
        config=config if is_valid else None
    )

async def _test_external_services(messages: List[str]) -> None:
    """
    Test connectivity to external services (Pinecone and Vertex AI).
    
    Args:
        messages: List to append error messages to
    """
    # Test Pinecone connectivity
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        await asyncio.sleep(0.1)  # Small delay to prevent rate limiting
        pc.list_indexes()  # This will fail if credentials are invalid
        logger.info("Successfully connected to Pinecone")
    except Exception as e:
        messages.append(f"Failed to connect to Pinecone: {str(e)}")

    # Test Vertex AI credentials
    try:
        import vertexai
        vertexai.init(
            project=os.getenv('VERTEX_PROJECT_ID'),
            location=os.getenv('VERTEX_LOCATION', 'europe-west2')
        )
        logger.info("Successfully initialized Vertex AI")
    except Exception as e:
        messages.append(f"Failed to initialize Vertex AI: {str(e)}")

async def process_directory(
    agent: TrademarkCaseAgent,
    directory: Path,
    batch_size: Optional[int] = None,
    max_retries: int = 3
) -> None:
    """
    Process all PDF files in the specified directory with enhanced error handling
    and retry logic.

    Args:
        agent: Initialized TrademarkCaseAgent
        directory: Path to directory containing PDF files
        batch_size: Optional limit on concurrent processing
        max_retries: Maximum number of retry attempts for failed processing
    """
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Track processing statistics
    stats = {
        'total': len(pdf_files),
        'successful': 0,
        'failed': 0,
        'retried': 0
    }

    # Process files concurrently in batches with retry logic
    for i in range(0, len(pdf_files), batch_size or 5):
        batch = pdf_files[i:i + (batch_size or 5)]
        batch_tasks = []
        
        for pdf_file in batch:
            task = asyncio.create_task(
                _process_with_retry(
                    agent,
                    pdf_file,
                    max_retries,
                    stats
                )
            )
            batch_tasks.append(task)
        
        # Wait for all tasks in the batch to complete
        await asyncio.gather(*batch_tasks)
        
        logger.info(
            f"Completed batch {i // (batch_size or 5) + 1}. "
            f"Progress: {stats['successful']}/{stats['total']} "
            f"(Failed: {stats['failed']}, Retried: {stats['retried']})"
        )

async def _process_with_retry(
    agent: TrademarkCaseAgent,
    pdf_file: Path,
    max_retries: int,
    stats: dict
) -> None:
    """
    Process a single PDF file with retry logic.

    Args:
        agent: TrademarkCaseAgent instance
        pdf_file: Path to the PDF file
        max_retries: Maximum number of retry attempts
        stats: Dictionary to track processing statistics
    """
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            await agent.process_pdf_streaming(str(pdf_file))
            stats['successful'] += 1
            if retry_count > 0:
                stats['retried'] += 1
            return
            
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                wait_time = min(2 ** retry_count, 30)  # Exponential backoff
                logger.warning(
                    f"Attempt {retry_count}/{max_retries} failed for {pdf_file.name}: "
                    f"{str(e)}. Retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"Failed to process {pdf_file.name} after {max_retries} attempts: {str(e)}"
                )
                stats['failed'] += 1

async def main():
    """
    Main async function to run the trademark processing pipeline with enhanced
    error handling and resource management.
    """
    parser = argparse.ArgumentParser(
        description="Process trademark case documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for failed processing"
    )
    args = parser.parse_args()

    start_time = datetime.now()
    agent = None

    try:
        # Load environment variables
        load_dotenv()
        
        # Verify environment and configuration
        validation_result = await verify_environment()
        
        if not validation_result.is_valid:
            for message in validation_result.messages:
                logger.error(message)
            raise ConfigurationError("Environment validation failed")
            
        for warning in validation_result.warnings:
            logger.warning(warning)
            
        config = validation_result.config

        # Create processing configuration
        processing_config = ProcessingConfig(
            batch_size=args.batch_size,
            chunk_size=config["CHUNK_SIZE"],
            chunk_overlap=config["CHUNK_OVERLAP"],
            max_retries=args.max_retries,
            wipo_confidence_threshold=config["WIPO_CONFIDENCE_THRESHOLD"],
            chunk_embedding_dimension=config["CHUNK_EMBEDDING_DIMENSION"],
            metadata_embedding_dimension=config["METADATA_EMBEDDING_DIMENSION"]
        )
        
        # Create Vertex AI configuration
        vertex_ai_config = VertexAIConfig.from_env()

        # Initialize agent with configurations
        agent = TrademarkCaseAgent(
            config=config,
            processing_config=processing_config,
            vertex_ai_config=vertex_ai_config
        )

        # Process PDF directory
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        await process_directory(
            agent,
            pdf_dir,
            batch_size=args.batch_size,
            max_retries=args.max_retries
        )

        # Log completion statistics
        duration = datetime.now() - start_time
        logger.info(
            f"Processing pipeline completed successfully in {duration}. "
            f"Check the log file for detailed statistics."
        )

    except Exception as e:
        logger.error(f"Processing pipeline failed: {str(e)}", exc_info=True)
        raise

    finally:
        # Ensure proper cleanup of resources
        if agent:
            try:
                await agent.cleanup()
                logger.info("Successfully cleaned up resources")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)