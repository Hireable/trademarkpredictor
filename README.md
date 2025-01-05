# Trademark Case AI Agent

## Overview

This project leverages AI and cloud technologies to process semi-structured trademark dispute case outcomes, extracting predictive data for further analysis. The extracted data is stored as vector embeddings in Pinecone, facilitating predictive machine learning (ML) and generative AI applications, such as predicting the outcomes of pending trademark cases.

## Features

- **PDF Processing**: Extracts text from PDF documents using Google Document AI.
- **Data Structuring**: Maps extracted text to a predefined schema for standardisation.
- **Vector Embedding**: Converts case data into vector embeddings using Legal-BERT for semantic analysis.
- **Storage in Pinecone**: Stores embeddings and associated metadata in Pinecone for efficient similarity searches and predictions.
- **Predictive Modelling**: Provides groundwork for predictive ML models by organising case data.

## Architecture

1. **Document Upload**: Users upload a PDF of a trademark dispute case. The file is stored in a Google Cloud Storage bucket.
2. **Text Extraction**: The uploaded PDF is processed by Google Document AI, extracting and chunking text for embedding.
3. **Vectorisation**: Chunked text is transformed into vector embeddings using Legal-BERT.
4. **Metadata Mapping**: Extracted data is structured according to a JSON schema.
5. **Storage**: Vectors and metadata are stored in a Pinecone index.
6. **Future Applications**: The dataset can be used for ML-based predictions and generative AI applications.

## Requirements

### Python Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- `google-cloud-storage`
- `google-cloud-documentai`
- `pinecone-client`
- `transformers`
- `torch`
- `pytest` (for testing)

### Cloud Services

1. **Google Cloud**:
   - **Document AI**: For PDF text extraction.
   - **Storage**: For storing raw and processed documents.
2. **Pinecone**:
   - Vector database for storing embeddings.

### Model

- **Legal-BERT**: Pre-trained model for embedding generation.

## Configuration

The application requires a configuration file (`config.py`) with the following settings:

- `PROJECT_ID`: Google Cloud project ID.
- `INDEX_NAME`: Name of the Pinecone index.
- `DIMENSION`: Vector embedding dimensions (e.g., 768).
- `SCHEMA_PATH`: Path to the JSON schema.
- Error message templates for debugging.

## Usage

### Processing a Case

1. Upload a trademark dispute PDF document.
2. Extract text using `TrademarkDocumentProcessor`.
3. Generate embeddings using `TrademarkCaseAgent`.
4. Store the vectors and metadata in Pinecone.
5. Query Pinecone for predictions or similarity analysis.

### Running Tests

Unit tests are available in the `tests` directory. Use `pytest` to run the test suite:

```bash
pytest
```

## Schema

The case data adheres to a predefined JSON schema (`tmpredictorschema.json`). Key fields include:

- `case_metadata`: Metadata such as case reference, decision date, and jurisdiction.
- `party_info`: Applicant and opponent details.
- `commercial_context`: Information on goods/services and market characteristics.
- `similarity_assessment`: Similarity metrics between trademarks.
- `outcome`: Decision and confidence score.

## Future Work

- **Predictive Modelling**: Train ML models on the Pinecone dataset to predict case outcomes.
- **Generative AI**: Leverage embeddings for generating case summaries and insights.
- **Scalability**: Enhance processing speed and add support for more jurisdictions.

## Contributions

Contributions are welcome! Please open an issue or submit a pull request to discuss proposed changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For queries or feedback, please reach out to joe\@gethireable.com.

