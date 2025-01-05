from google.cloud import storage
from google.cloud import documentai
import json
import os

class TrademarkDocumentProcessor:
    """
    Processes trademark documents through our analysis pipeline.
    This handles everything from upload to structured data extraction.
    """

    def __init__(self, project_id):
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.documentai_client = documentai.DocumentUnderstandingServiceClient()

    def upload_to_bucket(self, bucket_name, file_path):
        """
        Uploads a file to a Google Cloud Storage bucket.

        Args:
            bucket_name (str): The name of the target GCS bucket.
            file_path (str): The local file path to upload.

        Returns:
            str: GCS URI of the uploaded file.
        """
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        return f"gs://{bucket_name}/{blob.name}"

    def extract_text_from_pdf(self, gcs_uri):
        """
        Extracts text from a PDF stored in Google Cloud Storage using Document AI.

        Args:
            gcs_uri (str): The GCS URI of the PDF.

        Returns:
            str: Extracted text content.
        """
        input_config = {
            "gcs_source": {"uri": gcs_uri},
            "mime_type": "application/pdf",
        }

        request = {
            "input_config": input_config,
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }

        response = self.documentai_client.process_document(request=request)
        return response.text_annotations.text

    def chunk_text(self, text, chunk_size=512, overlap=50):
        """
        Splits text into chunks suitable for embedding generation.

        Args:
            text (str): The input text to split.
            chunk_size (int): The maximum size of each chunk (default: 512 tokens).
            overlap (int): The number of overlapping tokens between chunks (default: 50 tokens).

        Returns:
            list: A list of text chunks.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

# Example usage
if __name__ == "__main__":
    processor = TrademarkDocumentProcessor(project_id="your-gcp-project-id")
    gcs_uri = processor.upload_to_bucket("your-bucket-name", "path/to/pdf")
    extracted_text = processor.extract_text_from_pdf(gcs_uri)
    chunks = processor.chunk_text(extracted_text)
    print("Generated Chunks:", chunks)