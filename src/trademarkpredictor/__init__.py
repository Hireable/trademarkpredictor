import os
from trademarkpredictor.document_processor import TrademarkDocumentProcessor
from trademarkpredictor.trademark_analyzer import TrademarkAnalyzer
from trademarkpredictor.trademark_schema_extractor import TrademarkVectorSystem
from trademarkpredictor.config import get_config


def main():
    # Load configuration
    config = get_config()

    # Initialize components
    document_processor = TrademarkDocumentProcessor(project_id=config["PROJECT_ID"])
    analyzer = TrademarkAnalyzer(
        project_id=config["PROJECT_ID"],
        processor_id="your-processor-id",
        schema_path=config["SCHEMA_PATH"],
    )
    vector_system = TrademarkVectorSystem(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment="your-pinecone-environment",
    )

    # PDF upload and chunking
    file_path = "path/to/your/pdf"
    bucket_name = "your-gcs-bucket"
    gcs_uri = document_processor.upload_to_bucket(bucket_name, file_path)
    print(f"Uploaded to GCS: {gcs_uri}")

    with open(file_path, "rb") as pdf_file:
        chunks = document_processor.chunk_pdf(pdf_file.read())

    # Process chunks and store vectors
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        extracted_data = analyzer.process_chunk(chunk)

        # Validate data
        if analyzer.validate_against_schema(extracted_data):
            embedding = vector_system.generate_embedding(extracted_data["text"])
            vector_system.store_embedding(
                case_id=f"{os.path.basename(file_path)}_chunk_{i}",
                embedding=embedding,
                metadata=extracted_data,
            )
            print(f"Chunk {i+1} stored successfully.")
        else:
            print(f"Chunk {i+1} failed validation.")

    print("End-to-end process completed!")


if __name__ == "__main__":
    main()
