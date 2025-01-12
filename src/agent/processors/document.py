"""
Document processing utilities for handling PDFs and text chunking.
"""

import logging
from typing import List, Generator, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from unstructured.partition.pdf import partition_pdf
from src.exceptions import ProcessingError
from src.config.base import ProcessingConfig

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document text with metadata."""
    text: str
    page_number: Optional[int] = None
    chunk_number: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

class DocumentProcessor:
    """Handles document processing and chunking operations."""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize the document processor.

        Args:
            config: Processing configuration containing chunk size and overlap settings
        """
        self.config = config

    async def process_pdf(
        self, 
        file_path: str,
        include_metadata: bool = True
    ) -> Generator[DocumentChunk, None, None]:
        """
        Process a PDF file and yield chunks of text with metadata.

        Args:
            file_path: Path to the PDF file
            include_metadata: Whether to include metadata in chunks

        Yields:
            DocumentChunk objects containing text and metadata

        Raises:
            ProcessingError: If PDF processing fails
        """
        try:
            elements = partition_pdf(
                filename=file_path,
                strategy='fast',
                include_metadata=include_metadata
            )
            
            yield from self._chunk_elements(elements, Path(file_path).name)
            
        except Exception as e:
            raise ProcessingError(f"Failed to process PDF {file_path}: {str(e)}")

    def _chunk_elements(
        self,
        elements: List[Any],
        source_file: str
    ) -> Generator[DocumentChunk, None, None]:
        """
        Chunk document elements according to configuration.

        Args:
            elements: List of document elements from unstructured
            source_file: Name of source file for metadata

        Yields:
            DocumentChunk objects
        """
        current_chunk = []
        current_size = 0
        chunk_number = 0
        current_page = None

        for element in elements:
            # Extract page number if available
            if hasattr(element, 'metadata'):
                page_num = element.metadata.get('page_number')
                if page_num is not None:
                    current_page = page_num

            element_text = str(element)
            element_size = len(element_text)

            # If single element is larger than chunk size, split it
            if element_size > self.config.chunk_size:
                if current_chunk:
                    # Yield accumulated chunk first
                    yield self._create_chunk(
                        current_chunk,
                        chunk_number,
                        current_page,
                        source_file
                    )
                    chunk_number += 1
                    current_chunk = []
                    current_size = 0

                # Split large element into chunks
                words = element_text.split()
                temp_chunk = []
                temp_size = 0

                for word in words:
                    word_size = len(word) + 1  # +1 for space
                    if temp_size + word_size > self.config.chunk_size:
                        yield self._create_chunk(
                            temp_chunk,
                            chunk_number,
                            current_page,
                            source_file
                        )
                        chunk_number += 1
                        temp_chunk = []
                        temp_size = 0
                    
                    temp_chunk.append(word)
                    temp_size += word_size

                if temp_chunk:
                    yield self._create_chunk(
                        temp_chunk,
                        chunk_number,
                        current_page,
                        source_file
                    )
                    chunk_number += 1

            # Normal case: accumulate elements into chunks
            elif current_size + element_size > self.config.chunk_size:
                # Yield current chunk
                yield self._create_chunk(
                    current_chunk,
                    chunk_number,
                    current_page,
                    source_file
                )
                chunk_number += 1
                
                # Start new chunk with overlap
                if self.config.chunk_overlap > 0 and current_chunk:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = [overlap_text, element_text]
                    current_size = len(overlap_text) + element_size
                else:
                    current_chunk = [element_text]
                    current_size = element_size
            else:
                current_chunk.append(element_text)
                current_size += element_size

        # Yield final chunk if any remains
        if current_chunk:
            yield self._create_chunk(
                current_chunk,
                chunk_number,
                current_page,
                source_file
            )

    def _create_chunk(
        self,
        chunk_texts: List[str],
        chunk_number: int,
        page_number: Optional[int],
        source_file: str
    ) -> DocumentChunk:
        """
        Create a DocumentChunk from the given texts and metadata.

        Args:
            chunk_texts: List of text strings to combine
            chunk_number: Number of this chunk
            page_number: Page number if available
            source_file: Source file name

        Returns:
            DocumentChunk object
        """
        return DocumentChunk(
            text="\n\n".join(chunk_texts),
            page_number=page_number,
            chunk_number=chunk_number,
            metadata={
                "source_file": source_file,
                "chunk_number": chunk_number,
                "page_number": page_number
            }
        )

    def _get_overlap_text(self, chunk_texts: List[str]) -> str:
        """
        Get overlap text from the end of current chunk.

        Args:
            chunk_texts: List of text strings from current chunk

        Returns:
            String containing overlap text
        """
        combined_text = " ".join(chunk_texts)
        words = combined_text.split()
        
        # Calculate number of words for overlap
        overlap_word_count = max(
            1,
            int(self.config.chunk_overlap / 5)  # Approximate words based on chars
        )
        
        return " ".join(words[-overlap_word_count:])

    @staticmethod
    def extract_metadata(element: Any) -> Dict[str, Any]:
        """
        Extract metadata from a document element.

        Args:
            element: Document element from unstructured

        Returns:
            Dictionary of metadata
        """
        metadata = {}
        if hasattr(element, 'metadata'):
            metadata = {
                k: v for k, v in element.metadata.items()
                if k in ['page_number', 'filename', 'filetype']
            }
        return metadata