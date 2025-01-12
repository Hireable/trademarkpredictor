from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def validate_date_format(date_str: str) -> bool:
    """
    Validate if a date string matches the DD/MM/YYYY format.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r"^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$"
    if not re.match(pattern, date_str):
        return False
    
    try:
        datetime.strptime(date_str, "%d/%m/%Y")
        return True
    except ValueError:
        return False

def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing line endings.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()

def extract_nice_classes(text: str) -> List[int]:
    """
    Extract NICE classification numbers from text.
    
    Args:
        text: Text containing NICE class references
        
    Returns:
        List of unique NICE class numbers
    """
    # Look for patterns like "Class 9" or "Classes 9, 35 and 42"
    pattern = r'Class(?:es)?\s+(\d+(?:\s*,\s*\d+)*(?:\s+and\s+\d+)?)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if not matches:
        return []
    
    # Extract all numbers from matches
    classes = []
    for match in matches:
        numbers = re.findall(r'\d+', match)
        classes.extend(int(num) for num in numbers if 1 <= int(num) <= 45)
    
    return sorted(list(set(classes)))

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: Optional[int] = None
) -> List[str]:
    """
    Split text into overlapping chunks of specified size.
    
    Args:
        text: Text to split
        chunk_size: Target size for each chunk
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum allowed chunk size
        
    Returns:
        List of text chunks
    """
    if min_chunk_size is None:
        min_chunk_size = chunk_size // 2
    
    # Clean the text first
    text = clean_text(text)
    
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the current chunk
        end = start + chunk_size
        
        if end >= len(text):
            # If this is the last chunk and it's too small,
            # combine it with the previous chunk
            if len(text) - start < min_chunk_size and chunks:
                chunks[-1] = chunks[-1] + " " + text[start:]
            else:
                chunks.append(text[start:])
            break
        
        # Try to find a sentence end near the chunk boundary
        sentence_end = max(
            text.rfind(". ", start, end),
            text.rfind("? ", start, end),
            text.rfind("! ", start, end)
        )
        
        if sentence_end > start + min_chunk_size:
            # Add 2 to include the period and space
            end = sentence_end + 2
        else:
            # If no sentence end found, try to break at a space
            space = text.rfind(" ", start + min_chunk_size, end)
            if space != -1:
                end = space + 1
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

def safe_file_path(base_dir: Union[str, Path], filename: str) -> Path:
    """
    Create a safe file path, ensuring the directory exists and the path is secure.
    
    Args:
        base_dir: Base directory for the file
        filename: Name of the file
        
    Returns:
        Path object representing the safe file path
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove any path traversal attempts
    safe_name = Path(filename).name
    
    return base_dir / safe_name

def merge_dicts_with_lists(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries, concatenating lists for matching keys.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], list) and isinstance(value, list):
                result[key].extend(value)
            else:
                result[key] = value
        else:
            result[key] = value
    
    return result