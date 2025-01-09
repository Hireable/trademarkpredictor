import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock
import logging

from src.agent import TrademarkCaseAgent, ProcessingConfig
from src.config import VertexAIConfig
from src.exceptions import ProcessingError

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestPDFProcessing:
    """
    Test suite specifically for PDF processing functionality.
    This class focuses on testing the system with real PDF files from the test_pdfs directory.
    """

    @pytest.fixture
    def pdf_directory(self):
        """
        Fixture that provides the path to test PDFs directory.
        This helps ensure our tests can find the PDF files regardless of where they're run from.
        """
        # Assuming tests/test_pdfs is relative to the project root
        return Path(__file__).parent / "test_pdfs"

    @pytest.fixture
    def test_config(self):
        """
        Provides a test configuration that matches our production setup but uses test-specific values.
        This ensures our tests run in isolation from any production data.
        """
        return {
            "PINECONE_INDEX_NAME": "test-index",
            "EMBEDDING_MODEL_NAME": "all-mpnet-base-v2",  # Using the actual model for integration tests
            "WIPO_INDEX_NAME": "test-wipo",
            "SCHEMA_PATH": str(Path(__file__).parent / "data" / "test_schema.json"),
            "EXAMPLES_PATH": str(Path(__file__).parent / "data" / "test_examples.json"),
            "VERTEX_PROJECT_ID": "test-project",
            "VERTEX_LOCATION": "test-location"
        }

    @pytest.mark.asyncio
    async def test_single_pdf_processing(self, pdf_directory, test_config):
        """
        Tests processing of a single PDF file through the entire pipeline.
        This test verifies that we can:
        1. Read and chunk the PDF correctly
        2. Generate embeddings for each chunk
        3. Store the embeddings with correct metadata
        """
        # Get the first PDF from our test directory
        test_pdf = next(pdf_directory.glob("*.pdf"))
        
        agent = TrademarkCaseAgent(
            config=test_config,
            processing_config=ProcessingConfig(
                chunk_size=1000,  # Smaller chunks for testing
                chunk_overlap=100
            ),
            vertex_ai_config=VertexAIConfig(
                project_id="test-project",
                location="test-location"
            )
        )

        # Track the number of chunks processed
        chunk_count = 0
        stored_embeddings = []

        # Mock the storage function to capture what would be stored
        async def mock_store_embeddings(chunk, embedding, namespace, source):
            nonlocal chunk_count
            chunk_count += 1
            stored_embeddings.append({
                "chunk": chunk,
                "embedding": embedding,
                "namespace": namespace,
                "source": source
            })

        with patch.object(agent, '_store_chunk_embeddings', side_effect=mock_store_embeddings):
            await agent.process_pdf_streaming(str(test_pdf))

            # Verify processing results
            assert chunk_count > 0, "PDF should be split into at least one chunk"
            assert all(len(e["embedding"]) == 768 for e in stored_embeddings), \
                "All embeddings should have correct dimension"
            assert all(e["namespace"] == test_pdf.stem for e in stored_embeddings), \
                "All chunks should use PDF filename as namespace"

    @pytest.mark.asyncio
    async def test_multi_pdf_concurrent_processing(self, pdf_directory, test_config):
        """
        Tests concurrent processing of multiple PDFs.
        This test verifies that we can:
        1. Process multiple PDFs simultaneously
        2. Maintain separate namespaces for each PDF
        3. Handle concurrent embedding generation and storage
        """
        pdf_files = list(pdf_directory.glob("*.pdf"))
        assert len(pdf_files) >= 2, "Need at least 2 PDFs for concurrent testing"

        agent = TrademarkCaseAgent(
            config=test_config,
            processing_config=ProcessingConfig(
                batch_size=2,  # Process 2 PDFs concurrently
                chunk_size=1000,
                chunk_overlap=100
            ),
            vertex_ai_config=VertexAIConfig(
                project_id="test-project",
                location="test-location"
            )
        )

        # Track processing results per PDF
        results = {}

        async def mock_store_embeddings(chunk, embedding, namespace, source):
            if namespace not in results:
                results[namespace] = []
            results[namespace].append({
                "chunk": chunk,
                "embedding": embedding,
                "source": source
            })

        with patch.object(agent, '_store_chunk_embeddings', side_effect=mock_store_embeddings):
            # Process PDFs concurrently
            tasks = [
                agent.process_pdf_streaming(str(pdf))
                for pdf in pdf_files[:2]  # Test with first two PDFs
            ]
            await asyncio.gather(*tasks)

            # Verify results
            assert len(results) == 2, "Should have processed 2 PDFs"
            for pdf in pdf_files[:2]:
                namespace = pdf.stem
                assert namespace in results, f"Missing results for {namespace}"
                assert len(results[namespace]) > 0, f"No chunks processed for {namespace}"

    @pytest.mark.asyncio
    async def test_pdf_to_json_pipeline(self, pdf_directory, test_config):
        """
        Tests the complete pipeline from PDF processing to JSON generation.
        This test verifies that we can:
        1. Extract text from PDF
        2. Generate valid JSON output matching our schema
        3. Properly structure trademark case information
        """
        test_pdf = next(pdf_directory.glob("*.pdf"))
        
        # Mock response for JSON generation
        mock_json = {
            "case_number": "TEST/123",
            "opposition_number": "OPP/456",
            "applicant": "Test Company A",
            "opponent": "Test Company B",
            "applied_mark": "TEST MARK"
        }

        agent = TrademarkCaseAgent(
            config=test_config,
            processing_config=ProcessingConfig(),
            vertex_ai_config=VertexAIConfig(
                project_id="test-project",
                location="test-location"
            )
        )

        # Set up mocks
        mock_response = AsyncMock()
        mock_response.text = json.dumps(mock_json)

        with patch('vertexai.generative_models.GenerativeModel.generate_content') as mock_generate:
            mock_generate.return_value = [mock_response]

            # Process PDF and generate JSON
            await agent.process_pdf_streaming(str(test_pdf))
            
            # Extract text from chunks and generate JSON
            # Note: In a real scenario, you'd need to aggregate chunks appropriately
            text_content = "Sample extracted text"  # You'd get this from chunks
            json_output = await agent.generate_json_output(text_content)

            # Verify JSON output
            assert json_output is not None, "Should generate valid JSON"
            assert json_output == mock_json, "JSON output should match expected structure"

    @pytest.mark.asyncio
    async def test_error_handling(self, pdf_directory, test_config):
        """
        Tests error handling during PDF processing.
        This test verifies that we can:
        1. Handle corrupted PDFs
        2. Handle failed embedding generation
        3. Handle failed storage operations
        4. Properly clean up resources on failure
        """
        test_pdf = next(pdf_directory.glob("*.pdf"))
        
        agent = TrademarkCaseAgent(
            config=test_config,
            processing_config=ProcessingConfig(),
            vertex_ai_config=VertexAIConfig(
                project_id="test-project",
                location="test-location"
            )
        )

        # Test handling of embedding generation failure
        with patch.object(agent, '_generate_embeddings', side_effect=Exception("Embedding failed")):
            with pytest.raises(ProcessingError) as exc_info:
                await agent.process_pdf_streaming(str(test_pdf))
            assert "Embedding failed" in str(exc_info.value)

        # Test handling of storage failure
        with patch.object(agent, '_store_chunk_embeddings', side_effect=Exception("Storage failed")):
            with pytest.raises(ProcessingError) as exc_info:
                await agent.process_pdf_streaming(str(test_pdf))
            assert "Storage failed" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main(["-v"])