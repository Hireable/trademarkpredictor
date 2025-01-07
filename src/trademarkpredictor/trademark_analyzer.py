from google.cloud import documentai
import json


class TrademarkAnalyzer:
    """
    Processes PDF chunks and extracts structured data using Document AI.
    """

    def __init__(self, project_id, processor_id, schema_path):
        self.project_id = project_id
        self.processor_id = processor_id
        self.schema_path = schema_path
        self.docai_client = documentai.DocumentProcessorServiceClient()

    def process_chunk(self, chunk: bytes) -> dict:
        """
        Extracts structured data from a PDF chunk.

        Args:
            chunk (bytes): PDF chunk content.

        Returns:
            dict: Extracted structured data.
        """
        try:
            raw_document = documentai.RawDocument(content=chunk, mime_type="application/pdf")
            request = documentai.ProcessRequest(name=self.processor_id, raw_document=raw_document)
            response = self.docai_client.process_document(request=request)
            return {"text": response.document.text}
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return {}

    def validate_against_schema(self, data: dict) -> bool:
        """
        Validates extracted data against the provided schema.

        Args:
            data (dict): Extracted data.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            with open(self.schema_path, "r") as schema_file:
                schema = json.load(schema_file)
            from jsonschema import validate
            validate(instance=data, schema=schema)
            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False
