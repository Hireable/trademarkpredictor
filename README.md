# Trademark Predictor

## Overview

The **Trademark Predictor** project is an advanced AI-powered system designed to process and analyse trademark dispute cases. By leveraging cutting-edge AI technologies, this system extracts structured data from legal documents, generates vector embeddings using Legal-BERT, and performs predictive analysis with tools like Pinecone. The project also integrates **smolagents**, a novel library by Hugging Face, to create modular, intelligent agents for streamlined workflows.

## Key Features

1. **PDF Processing**:
   - Utilises Google Document AI for robust text extraction from legal PDFs.
2. **Schema Mapping**:
   - Transforms extracted text into a standardised schema based on a predefined JSON structure.
3. **Embedding Generation**:
   - Leverages **Legal-BERT**, a domain-specific model, for semantic embeddings.
4. **Vector Database Integration**:
   - Stores embeddings and metadata in Pinecone for efficient similarity searches.
5. **Predictive Modelling**:
   - Builds a foundation for ML-based predictions of case outcomes.
6. **Agent-Driven Workflows**:
   - Uses **smolagents** to automate and validate key steps in the processing pipeline.

## Vision

The ultimate goal of Trademark Predictor is to enable legal professionals, researchers, and organisations to:
- Anticipate trademark dispute outcomes.
- Analyse historical cases for trends and insights.
- Accelerate decision-making through data-driven recommendations.

## Architecture

### Workflow

1. **Document Input**:
   - Users upload trademark dispute case PDFs.
2. **Text Extraction**:
   - Documents are processed using Google Document AI.
3. **Data Structuring**:
   - Extracted text is mapped to the JSON schema (`tmpredictorschema.json`).
4. **Embedding Generation**:
   - Text is converted to vector embeddings using Legal-BERT.
5. **Agent Automation**:
   - **smolagents** manage tasks such as validation, processing, and database interactions.
6. **Storage and Querying**:
   - Vectors and metadata are stored in Pinecone, enabling advanced similarity searches.

### Tools and Libraries

- **smolagents**: Orchestrates modular AI agent tasks.
- **Legal-BERT**: A pre-trained transformer model fine-tuned for legal documents.
- **Pinecone**: A vector database for efficient data retrieval.
- **Google Document AI**: Extracts structured data from unstructured PDFs.

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud API key
- Pinecone API key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/trademark-predictor.git
   cd trademark-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Set up API keys for Pinecone and Google Cloud.

### Configuration

Update the `config.py` file with your project-specific details:
- Google Cloud project ID
- Pinecone index name and API key
- JSON schema path

## Usage

### Processing Documents

1. Upload a trademark case PDF.
2. Trigger processing via the agent (`TrademarkCaseAgent`):
   ```python
   from test_agent import TrademarkCaseAgent

   agent = TrademarkCaseAgent()
   agent.process_case(case_data)
   ```
3. Query processed data in Pinecone for similarity analysis or predictions.

### Running Tests

Run unit tests to verify the pipeline:
```bash
pytest
```

## Future Enhancements

- **Generative AI**: Use embeddings to generate case summaries and insights.
- **Cross-Jurisdictional Analysis**: Support for additional jurisdictions.
- **Real-Time Predictions**: Provide dynamic recommendations based on incoming data.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request for review.

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.

## Contact

For queries or suggestions, please email **joe@gethireable.com**.
