from setuptools import setup, find_packages
from ..src.agent import TrademarkCaseAgent, ProcessingConfig

setup(
    name="trademarkpredictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",  # For running your test suite
        "pytest-asyncio",  # Required for async test support (you're using @pytest.mark.asyncio)
        "google-cloud-aiplatform",  # For Vertex AI integration
        "vertexai",  # For the generative models you're using
        "pinecone-client",  # Based on your PINECONE_INDEX_NAME config
        "sentence-transformers",  # For the all-mpnet-base-v2 embedding model
        "PyPDF2",  # A common choice for PDF processing, though you might be using a different PDF library
        "pydantic",  # Often used with ProcessingConfig-style classes
        "aiohttp",  # Common requirement for async operations
    ],
    extras_require={
        "dev": [
            "black",  # Code formatting
            "isort",  # Import sorting
            "flake8",  # Code linting
            "mypy",  # Type checking
            "pytest-cov",  # Test coverage reporting
        ],
        "test": [
            "pytest-mock",  # For the mocking functionality you're using
            "asynctest",  # Additional async testing utilities
        ]
    },
    python_requires=">=3.8",  # Based on async features usage
)