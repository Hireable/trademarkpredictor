from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1",
    packages=find_packages(),  # Changed this line
    package_dir={"": "src"},  # Added this line
    install_requires=[
        "pytest>=7.0.0",
        "pytest-asyncio",
        "google-cloud-aiplatform",
        "vertexai",
        "pinecone-client",
        "sentence-transformers",
        "PyPDF2",
        "pydantic",
        "aiohttp",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pytest-cov",
        ],
        "test": [
            "pytest-mock",
            "asynctest",
        ],
    },
    python_requires=">=3.8",
)