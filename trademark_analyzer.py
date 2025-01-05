from google.cloud import documentai, storage
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
import json

class TrademarkAnalyzer:
    def __init__(self, project_id, processor_id, location="europe-west2"):
        self.project_id = project_id
        self.processor_id = processor_id
        self.location = location
        self.docai_client = documentai.DocumentProcessorServiceClient()
        self.processor_path = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        self.storage_client = storage.Client(project=project_id)

    def split_pdf_into_chunks(self, pdf_content, chunk_size=15):
        """
        Splits a PDF into smaller chunks for processing.
        """
        try:
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

        except Exception as e:
            print(f"Error splitting PDF: {str(e)}")
            return None

    def process_document_chunk(self, chunk_content):
        """
        Processes a single PDF chunk using Document AI.
        """
        try:
            raw_document = documentai.RawDocument(content=chunk_content, mime_type='application/pdf')
            request = documentai.ProcessRequest(name=self.processor_path, raw_document=raw_document)
            result = self.docai_client.process_document(request=request)
            return self._organize_document_text(result.document)

        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return None

    def analyze_document(self, bucket_name, blob_name):
        """
        Analyzes a trademark document in chunks and combines the results.
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            content = blob.download_as_bytes()

            chunks = self.split_pdf_into_chunks(content)
            if not chunks:
                return None

            combined_result = {'full_text': '', 'pages': []}
            for chunk in chunks:
                chunk_result = self.process_document_chunk(chunk)
                if chunk_result:
                    combined_result['full_text'] += chunk_result['full_text'] + '\n'
                    combined_result['pages'].extend(chunk_result['pages'])

            return combined_result

        except Exception as e:
            print(f"Error analyzing document: {str(e)}")
            return None

    def _organize_document_text(self, document):
        """
        Organizes extracted text into a structured format.
        """
        organized_content = {'full_text': document.text, 'pages': []}

        for page in document.pages:
            page_content = {'page_number': page.page_number, 'paragraphs': []}
            for paragraph in page.paragraphs:
                para_text = self._get_text(paragraph.layout.text_anchor, document.text)
                page_content['paragraphs'].append(para_text)
            organized_content['pages'].append(page_content)

        return organized_content

    def _get_text(self, text_anchor, document_text):
        if not text_anchor.text_segments:
            return ''

        text = ''
        for segment in text_anchor.text_segments:
            text += document_text[segment.start_index:segment.end_index]
        return text.strip()

    def extract_to_json(self, structured_text, schema_path):
        """
        Extracts structured data into a JSON format using a predefined schema.
        """
        try:
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)

            # Map structured text to schema here (placeholder for logic)
            # Implement rules for extracting values based on keywords or patterns

            extracted_data = {
                # Placeholder for extraction logic
            }

            # Validate against schema
            from jsonschema import validate
            validate(instance=extracted_data, schema=schema)
            return extracted_data

        except Exception as e:
            print(f"Error extracting to JSON: {str(e)}")
            return None

if __name__ == "__main__":
    analyzer = TrademarkAnalyzer(
        project_id="trademark-case-agent",
        processor_id="306d8ce7e4dac069"
    )
    result = analyzer.analyze_document(
        bucket_name="trademark-case-agent-raw-documents",
        blob_name="sample-trademark-case.pdf"
    )
    if result:
        print(json.dumps(result, indent=2))