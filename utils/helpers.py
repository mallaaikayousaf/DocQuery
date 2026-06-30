"""
Utility helper functions for the RAG application.
"""


def format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining = seconds % 60
        return f"{minutes}m {remaining:.0f}s"
