# TrademarkPredictor

**TrademarkPredictor** is a sophisticated AI-powered tool designed to analyze and process trademark opposition decisions from PDF documents. It leverages advanced natural language processing (NLP) techniques, including **Retrieval-Augmented Generation (RAG)** and **Google Vertex AI's Gemini model**, to extract structured data from legal documents and predict outcomes. This tool is particularly useful for legal professionals, trademark attorneys, and businesses looking to streamline the analysis of trademark cases.

The project integrates multiple technologies, including:
- **Google Vertex AI** for generative text processing.
- **Pinecone** for vector storage and retrieval.
- **Sentence Transformers** for generating embeddings.
- **Unstructured** for PDF document parsing.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Workflow](#workflow)
5. [Configuration](#configuration)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

---

## Features

- **PDF Document Processing**: Efficiently processes large PDF documents using streaming to handle chunking and embedding generation.
- **Structured Data Extraction**: Extracts structured JSON data from trademark opposition decisions using a predefined schema.
- **Few-Shot Learning**: Incorporates few-shot examples to guide the AI in generating accurate and consistent outputs.
- **Vector Storage**: Uses Pinecone for storing and retrieving document embeddings, enabling efficient similarity searches.
- **Validation**: Ensures generated JSON outputs conform to a predefined schema for consistency and accuracy.
- **Concurrency**: Utilizes `asyncio` for concurrent processing, improving performance for large-scale operations.

---

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Hireable/trademarkpredictor.git
   cd trademarkpredictor
   ```

2. **Set Up a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add the following variables:
   ```plaintext
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   VERTEX_PROJECT_ID=your_vertex_ai_project_id
   VERTEX_LOCATION=your_vertex_ai_location
   EMBEDDING_MODEL_NAME=your_embedding_model_name
   SCHEMA_PATH=path/to/schema.json
   EXAMPLES_PATH=path/to/examples.json
   ```

5. **Download Pre-Trained Models**:
   Ensure the required models (e.g., Sentence Transformers) are downloaded and available.

---

## Usage

### Processing a PDF Document

To process a PDF document and extract structured data, use the following code:

```python
from agent import TrademarkCaseAgent

# Initialize the agent
agent = TrademarkCaseAgent()

# Process a PDF document
pdf_path = "path/to/your/document.pdf"
namespace = "custom_namespace"  # Optional namespace for vector storage

await agent.process_pdf_streaming(pdf_path, namespace)
```

### Generating JSON Output

To generate structured JSON output from a case text:

```python
case_text = "Your trademark case text here..."
json_output = await agent.generate_json_output(case_text)

print(json_output)
```

### Cleaning Up Resources

After processing, ensure resources are cleaned up:

```python
await agent.cleanup()
```

---

## Workflow

1. **Document Chunking**: The PDF document is split into smaller chunks for efficient processing.
2. **Embedding Generation**: Each chunk is converted into embeddings using a pre-trained Sentence Transformer model.
3. **Vector Storage**: Embeddings are stored in Pinecone for future retrieval.
4. **Text Generation**: The Gemini model generates structured JSON output based on the input text and few-shot examples.
5. **Validation**: The generated JSON is validated against a predefined schema to ensure accuracy and consistency.

---

## Configuration

The project is highly configurable. Key configuration options include:

- **ProcessingConfig**: Controls chunk size, overlap, batch size, and retry settings.
- **VertexAIConfig**: Manages Google Vertex AI settings, including project ID and location.
- **Schema and Examples**: Define the structure of the output JSON and provide few-shot examples for the AI.

Configuration is loaded from environment variables and JSON files. Ensure all required paths and keys are correctly set in the `.env` file.

---

## Contributing

We welcome contributions to improve TrademarkPredictor! Here’s how you can help:

1. **Report Issues**: If you find a bug or have a feature request, please open an issue on GitHub.
2. **Submit Pull Requests**: Fork the repository, make your changes, and submit a pull request.
3. **Improve Documentation**: Help us improve the README, code comments, or add tutorials.

Please read our [Contribution Guidelines](CONTRIBUTING.md) for more details.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Google Vertex AI**: For providing the Gemini model and generative AI capabilities.
- **Pinecone**: For enabling efficient vector storage and retrieval.
- **Sentence Transformers**: For generating high-quality embeddings.
- **Unstructured**: For simplifying PDF document parsing.

---

## Contact

For questions or feedback, please reach out to:
- **Joe Brown**: [joe@gethireable.com](mailto:joe@gethireable.com)
- **GitHub Issues**: [Open an Issue](https://github.com/Hireable/trademarkpredictor/issues)
