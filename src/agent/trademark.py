"""
Trademark-specific implementation of the document processing agent.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import aiohttp
from contextlib import asynccontextmanager

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from ..agent.base import BaseAgent
from ..exceptions import ProcessingError, ValidationError
from ..storage.pinecone import PineconeStorage
from ..config.schema import get_validator
from ..config.settings import get_settings
from ..agent.processors.document import DocumentProcessor
from ..agent.processors.json import JsonProcessor
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class TrademarkAgent(BaseAgent):
    """
    Agent for processing trademark opposition decisions and generating structured data.
    
    This class implements the abstract methods from BaseAgent with specific logic
    for trademark document processing.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the TrademarkAgent with additional trademark-specific setup."""
        super().__init__(*args, **kwargs)
        
        settings = get_settings()
        
        # Initialize processors
        self.document_processor = DocumentProcessor(self.processing_config)
        self.json_processor = JsonProcessor(
            schema_path=settings.schema_path,
            examples_path=settings.examples_path,
            vertex_project_id=settings.vertex_ai.project_id,
            vertex_location=settings.vertex_ai.location
        )
        
        # Initialize vector store
        self.storage = PineconeStorage(
            api_key=settings.pinecone.api_key,
            environment=settings.pinecone.environment,
            cloud=settings.pinecone.location,
            region=settings.vertex_ai.location
        )
        
        # Configure generation parameters
        self.generation_config = {
            "max_output_tokens": settings.vertex_ai.max_tokens,
            "temperature": settings.vertex_ai.temperature,
            "top_p": settings.vertex_ai.top_p,
            "response_mime_type": "application/json",
        }
        
        logger.info("TrademarkAgent initialized with processors and storage")
    
    @asynccontextmanager
    async def session_context(self):
        """Context manager for aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        try:
            yield self.session
        finally:
            if self.session:
                await self.session.close()
                self.session = None
    
    async def process_document(self, document_path: str, namespace: Optional[str] = None) -> None:
        """
        Process a trademark document using streaming for efficiency.
        
        Args:
            document_path: Path to the document to process
            namespace: Optional namespace for vector store
        """
        if namespace is None:
            namespace = Path(document_path).stem
        
        try:
            # Initialize storage if needed
            await self.storage.initialize(
                dimension=self.processing_config.embedding_dimension,
                index_name=self.config["PINECONE_INDEX_NAME"]
            )
            
            # Clean up existing namespace
            await self.storage.delete(delete_all=True, namespace=namespace)
            
            # Process document in chunks
            async for chunk in self.document_processor.process_pdf(document_path):
                embeddings = await self.generate_embeddings(chunk.text)
                await self.store_embeddings(
                    embeddings,
                    {
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "chunk_number": chunk.chunk_number,
                        "filename": Path(document_path).name,
                        "namespace": namespace,
                    }
                )
            
            logger.info(f"Successfully processed document: {document_path}")
            
        except Exception as e:
            logger.error(f"Failed to process document {document_path}: {str(e)}")
            raise ProcessingError(f"Failed to process document {document_path}: {str(e)}")
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text chunk."""
        try:
            return self.resources.embedding_model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise ProcessingError(f"Embedding generation failed: {str(e)}")
    
    async def store_embeddings(self, embeddings: list, metadata: Dict[str, Any]) -> None:
        """Store embeddings in the vector store."""
        try:
            chunk_id = f"{metadata['namespace']}-{hash(metadata['text'])}"
            await self.storage.upsert(
                vectors=[{
                    "id": chunk_id,
                    "values": embeddings,
                    "metadata": metadata
                }],
                namespace=metadata["namespace"]
            )
            logger.debug(f"Stored embeddings for chunk {chunk_id}")
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            raise ProcessingError(f"Failed to store embeddings: {str(e)}")
    
    async def generate_json_output(self, text_content: str) -> Optional[Dict]:
        """
        Generate structured JSON output from text content using Vertex AI.
        
        Args:
            text_content: The text content to process
        
        Returns:
            Dictionary containing the structured data, or None if generation fails
        """
        try:
            return await self.json_processor.generate_json(text_content)
        except Exception as e:
            logger.error(f"Failed to generate JSON output: {str(e)}")
            return None
    
    async def process_query(
        self,
        query_text: str,
        namespace: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process a query against stored documents.
        
        Args:
            query_text: Query text
            namespace: Optional namespace to search in
            top_k: Number of results to return
            
        Returns:
            List of matching results with scores
        """
        try:
            # Generate query embeddings
            query_embedding = await self.generate_embeddings(query_text)
            
            # Query vector store
            results = await self.storage.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            return [
                {
                    "text": vec.metadata["text"],
                    "score": vec.score,
                    "page_number": vec.metadata.get("page_number"),
                    "filename": vec.metadata.get("filename")
                }
                for vec in results.vectors
            ]
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise ProcessingError(f"Failed to process query: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        try:
            await super().cleanup()
            await self.storage.cleanup()
            logger.info("TrademarkAgent cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise ProcessingError(f"Cleanup failed: {str(e)}")