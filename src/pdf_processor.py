import io

from pypdf import PdfReader


def extract_text_from_path(pdf_path: str) -> str:

    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}") from exc
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}") from e


def extract_text_from_bytes(pdf_bytes: bytes) -> str:

    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        raise RuntimeError(f"Error reading PDF from bytes: {e}") from e


def get_page_count(pdf_path: str) -> int:

    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            return len(reader.pages)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}") from exc
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}") from e


def validate_pdf(pdf_bytes: bytes) -> bool:

    if not pdf_bytes.startswith(b"%PDF"):
        return False
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    if len(reader.pages) == 0:
        return False

    try:
        first_page = reader.pages[0]
        first_page.extract_text()
    except Exception:
        # Even if text extraction fails, it might still be a valid PDF
        # (e.g., scanned document with no text layer)
        pass

    return True