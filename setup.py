from setuptools import setup, find_packages
import os

# Read the contents of README.md for the long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies required for the project to function
core_requirements = [
    "google-cloud-aiplatform>=1.25.0",    # For Vertex AI integration
    "vertexai>=0.0.1",                    # For Gemini model access
    "pinecone-client>=2.2.1",             # For vector database operations
    "sentence-transformers>=2.2.2",        # For generating embeddings
    "unstructured[local-inference]>=0.6.8",# For PDF parsing
    "python-dotenv>=1.0.0",               # For environment variable management
    "pydantic>=2.0.0",                    # For data validation and settings management
    "pydantic-settings",
    "aiohttp>=3.8.0",                     # For async HTTP operations
    "tenacity>=8.0.0",                    # For retry logic
    "jsonschema>=4.0.0",                  # For JSON schema validation
    "pdfminer.six>=20221105",             # For PDF text extraction
]

# Development dependencies for testing, linting, etc.
dev_requirements = [
    "pytest>=7.0.0",                      # For running tests
    "pytest-asyncio>=0.20.0",             # For testing async code
    "pytest-cov>=4.0.0",                  # For test coverage reporting
    "black>=22.0.0",                      # For code formatting
    "isort>=5.0.0",                       # For import sorting
    "flake8>=4.0.0",                      # For code linting
    "mypy>=1.0.0",                        # For static type checking
    "pre-commit>=3.0.0",                  # For git pre-commit hooks
]

# Additional requirements for documentation
docs_requirements = [
    "sphinx>=4.0.0",                      # For documentation generation
    "sphinx-rtd-theme>=1.0.0",            # Documentation theme
    "sphinx-autodoc-typehints>=1.12.0",   # For type hint documentation
]

setup(
    name="trademarkpredictor",
    version="0.2.0",  # Using semantic versioning
    description="AI-powered trademark opposition decision analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Joe Brown",
    author_email="joe@gethireable.com",
    url="https://github.com/Hireable/trademarkpredictor",
    
    # Package configuration
    package_dir={"": "src"},              # Specify src directory as package root
    packages=find_packages(where="src"),   # Automatically find all packages in src/
    python_requires=">=3.8",              # Minimum Python version required
    
    # Dependencies
    install_requires=core_requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "all": dev_requirements + docs_requirements,
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "trademark-predict=trademarkpredictor.__main__:main",
        ],
    },
    
    # Project classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
    ],
    
    # Additional package metadata
    keywords="trademark, legal, ai, nlp, machine-learning",
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)