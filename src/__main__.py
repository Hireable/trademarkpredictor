import os
import logging
import argparse  # For command-line arguments
from dotenv import load_dotenv

from src.agent import TrademarkCaseAgent
from src.config import get_config

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Let's check our environment variables first
    print("\nEnvironment Variable Check:")
    api_key = os.getenv('PINECONE_API_KEY')
    env = os.getenv('PINECONE_ENVIRONMENT')
    print(f"API Key exists: {bool(api_key)}")
    print(f"API Key length (if exists): {len(api_key) if api_key else 0}")
    print(f"Environment exists: {bool(env)}")
    print(f"Environment value (if exists): {env if env else 'Not found'}")
    
    # Now let's check what's in our config
    print("\nConfiguration Check:")
    config = get_config()
    for key, value in config.items():
        if 'API_KEY' in key:
            print(f"{key}: {'[EXISTS]' if value else '[MISSING]'}")
        else:
            print(f"{key}: {value}")

    # Then try to create the agent
    agent = TrademarkCaseAgent()

    # Process PDFs

    pdf_dir = os.path.join("data", "raw_pdfs") # Assumed directory for raw PDFs
    for filename in os.listdir(pdf_dir):

        if filename.endswith(".pdf"):

            pdf_path = os.path.join(pdf_dir, filename)
            agent.process_pdf(pdf_path)

    # Load WIPO mapping
    agent.load_wipo_mapping()

    logger.info("PDF processing and WIPO mapping complete.")




if __name__ == "__main__":
    main()