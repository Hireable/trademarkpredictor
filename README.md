## Basic project overview

* **Project name**: Trademark Predictor
* **Description**: This project processes trademark case documents to extract structured data using AI models.
* **Technologies used**: Python, Vertex AI, Pinecone, Sentence Transformers, Unstructured, PDFMiner, OpenPyXL, Dotenv
* **Directory structure**:
  * `src/`: Contains the main source code
    * `src/__main__.py`: Entry point for the application
    * `src/agent.py`: Contains the `TrademarkCaseAgent` class for processing trademark cases
    * `src/config.py`: Configuration settings for the application
    * `src/exceptions.py`: Custom exceptions used in the project
    * `src/utils.py`: Utility functions for interacting with Pinecone
  * `data/`: Contains data files
    * `data/schema.json`: JSON schema for validating extracted data
    * `data/examples.json`: Few-shot examples for the AI model
  * `tests/`: Contains test files
  * `.env`: Environment variables
  * `requirements.txt`: List of dependencies
  * `.gitignore`: Files and directories to be ignored by Git

## Setup and installation

* **Prerequisites**: Ensure you have Python 3.8 or higher installed.
* **Clone the repository**: `git clone <repository-url>`
* **Navigate to the project directory**: `cd trademarkpredictor`
* **Create a virtual environment**: `python -m venv .venv`
* **Activate the virtual environment**:
  * On Windows: `.\.venv\Scripts\activate`
  * On macOS/Linux: `source .venv/bin/activate`
* **Install dependencies**: `pip install -r requirements.txt`
* **Set up environment variables**: Create a `.env` file in the root directory with the following content:
  ```plaintext
  PINECONE_API_KEY=<your-pinecone-api-key>
  PINECONE_ENVIRONMENT=<your-pinecone-environment>
  VERTEX_PROJECT_ID=<your-vertex-project-id>
  ```
* **Run the application**: `python -m src`

## Contributing guidelines

* **Fork the repository**: Click on the "Fork" button at the top right of the repository page.
* **Clone your fork**: `git clone <your-fork-url>`
* **Create a new branch**: `git checkout -b <branch-name>`
* **Make your changes**: Ensure your code follows the project's coding standards.
* **Run tests**: Ensure all tests pass by running `pytest`.
* **Commit your changes**: `git commit -m "Description of your changes"`
* **Push to your fork**: `git push origin <branch-name>`
* **Create a pull request**: Go to the original repository and click on "New Pull Request".
