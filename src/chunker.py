"""
Text Chunking Module
Splits large text into smaller, overlapping chunks.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
        
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 100
) -> List[str]:
   
    if not text or not isinstance(text, str):
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentences
            "? ",    # Question marks
            "! ",    # Exclamation marks
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Characters (last resort)
        ],
        add_start_index=True
    )
    chunks=text_splitter.split_text(text)
    return chunks


def get_chunk_statistics(chunks: List[str]) -> dict:

    if not chunks: 
        return {
            'count': 0,
            'avg_length': 0.0,
            'min_length': 0,
            'max_length': 0,
            'total_chars': 0,
            'chunks_by_size': {
                'small': 0,
                'medium': 0,
                'large': 0
            }
        }
    
    chunk_lengths = [len(chunk) for chunk in chunks]
    
    stats = {
        'count': len(chunks),
        'avg_length': sum(chunk_lengths) / len(chunks),
        'min_length': min(chunk_lengths) if chunk_lengths else 0,
        'max_length': max(chunk_lengths) if chunk_lengths else 0,
        'total_chars': sum(chunk_lengths),
        'chunks_by_size': {
            'small': len([c for c in chunk_lengths if c < 100]),
            'medium': len([c for c in chunk_lengths if 100 <= c < 400]),
            'large': len([c for c in chunk_lengths if c >= 400])
        }
    }
    
    return stats
