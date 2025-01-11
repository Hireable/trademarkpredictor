import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json
from PyPDF2 import PdfReader
from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"test_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles the complete document processing pipeline with detailed validation"""

    def __init__(self):
        """Initialize processor components"""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.case_index = self.pc.Index(os.getenv('PINECONE_CHUNKS_INDEX'))
        self.predictive_index = self.pc.Index(os.getenv('PINECONE_PREDICTIVE_INDEX'))

        # Initialize embedding model for 1024 dimensions
        logger.info("Initializing embedding model (multilingual-e5-large)")
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

        # Load schema for validation
        with open(os.getenv('SCHEMA_PATH', 'data/schema.json')) as f:
            self.schema = json.load(f)

    async def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Processing statistics and results
        """
        logger.info(f"\n{'='*80}\nStarting document processing: {pdf_path}\n{'='*80}")

        try:
            # Step 1: Load and validate document
            document_text = await self._load_document(pdf_path)
            logger.info(f"Document loaded successfully: {len(document_text)} characters")

            # Step 2: Create chunks
            chunks = await self._create_chunks(document_text)
            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Generate embeddings
            chunk_embeddings = await self._generate_embeddings(chunks)
            logger.info(f"Generated embeddings - Shape: {chunk_embeddings.shape}")

            # Step 4: Extract metadata
            metadata = await self._extract_metadata(document_text)
            logger.info("Extracted base metadata")

            # Step 5: Validate against schema
            if self._validate_metadata(metadata):
                logger.info("Metadata validation successful")

            # Step 6: Store in Pinecone
            storage_results = await self._store_vectors(
                chunks,
                chunk_embeddings,
                metadata,
                Path(pdf_path).stem
            )
            logger.info("Vector storage complete")

            # Return processing statistics
            return {
                'document_path': pdf_path,
                'num_chunks': len(chunks),
                'embedding_dimension': chunk_embeddings.shape[1],
                'storage_results': storage_results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise

    async def _load_document(self, pdf_path: str) -> str:
        """Load and validate PDF document"""
        logger.info("Loading document...")
        try:
            reader = PdfReader(pdf_path)
            content = " ".join(page.extract_text() for page in reader.pages if page.extract_text())

            if not content.strip():
                raise ValueError("Document is empty or could not extract text")

            return content
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise

    async def _create_chunks(self, text: str, chunk_size: int = 1500) -> List[str]:
        """Create text chunks with overlap"""
        logger.info("Creating document chunks...")

        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_size = len(sentence)

            if current_size + sentence_size > chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[-1])
                else:
                    chunks.append(sentence)
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

        chunk_sizes = [len(chunk) for chunk in chunks]
        logger.info(f"""
        Chunking Statistics:
        - Number of chunks: {len(chunks)}
        - Average chunk size: {sum(chunk_sizes)/len(chunks):.0f} characters
        - Max chunk size: {max(chunk_sizes)} characters
        - Min chunk size: {min(chunk_sizes)} characters
        """)

        return chunks

    async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        logger.info("Generating embeddings...")

        try:
            embeddings = self.embedding_model.encode(
                chunks,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            if embeddings.shape[1] != 1024:
                raise ValueError(f"Unexpected embedding dimension: {embeddings.shape[1]}")

            logger.info(f"""
            Embedding Statistics:
            - Shape: {embeddings.shape}
            - Mean: {embeddings.mean():.4f}
            - Std: {embeddings.std():.4f}
            """)

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    async def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract structured metadata from document text"""
        logger.info("Extracting metadata...")

        metadata = {
            "case_metadata": {
                "case_ref": "Sample/2025/001",
                "officer": "Test Officer",
                "dec_date": "2025-01-11",
                "jurisdiction": "UK"
            },
            "party_info": {
                "app_mark": "Test Mark",
                "opp_mark": "Prior Mark",
                "app_name": "Test Applicant",
                "opp_name": "Test Opponent"
            },
            "commercial_context": {
                "app_spec": "Sample goods and services",
                "opp_spec": "Sample goods and services",
                "app_class": [25],
                "opp_class": [25]
            },
            "similarity_assessment": {
                "mark_similarity": {
                    "vis_sim": 3,
                    "aur_sim": 3,
                    "con_sim": 3
                },
                "gds_sim": {
                    "nature": 4,
                    "purpose": 4,
                    "channels": 3,
                    "use": 3
                }
            },
            "outcome": {
                "confusion": True,
                "conf_type": "indirect"
            }
        }

        return metadata

    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata against schema"""
        try:
            required_keys = ['case_metadata', 'party_info', 'commercial_context', 'similarity_assessment', 'outcome']

            missing_keys = [key for key in required_keys if key not in metadata]

            if missing_keys:
                raise ValueError(f"Missing required metadata keys: {missing_keys}")

            return True

        except Exception as e:
            logger.error(f"Metadata validation failed: {str(e)}")
            raise

    async def _store_vectors(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        metadata: Dict[str, Any],
        namespace: str
    ) -> Dict[str, Any]:
        """Store vectors and metadata in Pinecone"""
        logger.info("Storing vectors in Pinecone...")

        try:
            chunk_vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_vectors.append({
                    'id': f'chunk_{i}',
                    'values': embedding.tolist(),
                    'metadata': {
                        'text': chunk,
                        'position': i,
                        **metadata
                    }
                })

            await self.case_index.upsert(
                vectors=chunk_vectors,
                namespace=namespace
            )

            return {
                "status": "success",
                "vectors_stored": len(chunk_vectors)
            }

        except Exception as e:
            logger.error(f"Vector storage failed: {str(e)}")
            raise
