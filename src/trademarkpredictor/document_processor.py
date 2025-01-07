from google.cloud import storage
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
import os


class TrademarkDocumentProcessor:
    """
    Handles file uploads, chunking, and storage for trademark documents.
    """

    def __init__(self, project_id):
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)

    def upload_to_bucket(self, bucket_name: str, file_path: str) -> str:
        """
        Uploads a file to Google Cloud Storage.

        Args:
            bucket_name (str): Name of the GCS bucket.
            file_path (str): Local path to the file.

        Returns:
            str: GCS URI of the uploaded file.
        """
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        return f"gs://{bucket_name}/{blob.name}"

    def chunk_pdf(self, pdf_content: bytes, chunk_size: int = 15) -> list:
        """
        Splits a PDF into smaller chunks for processing.

        Args:
            pdf_content (bytes): Content of the PDF.
            chunk_size (int): Number of pages per chunk.

        Returns:
            list: List of byte chunks.
        """
        pdf = PdfReader(BytesIO(pdf_content))
        total_pages = len(pdf.pages)
        chunks = []

        for start in range(0, total_pages, chunk_size):
            end = min(start + chunk_size, total_pages)
            writer = PdfWriter()

            for page_num in range(start, end):
                writer.add_page(pdf.pages[page_num])

            output = BytesIO()
            writer.write(output)
            chunks.append(output.getvalue())

        return chunks
