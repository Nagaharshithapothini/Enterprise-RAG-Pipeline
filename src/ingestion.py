from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf"
}


def load_text_file(file_path: str) -> List[Dict]:

    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8"
    )

    return [
        {
            "text": text,
            "metadata": {
                "source": path.name,
                "file_type": "txt"
            }
        }
    ]


def load_pdf_file(file_path: str) -> List[Dict]:

    path = Path(file_path)

    reader = PdfReader(
        file_path
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": path.name,
                    "file_type": "pdf",
                    "page": page_number
                }
            }
        )

    return documents


def load_document(
    file_path: str
) -> List[Dict]:

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".txt":
        return load_text_file(file_path)

    if extension == ".pdf":
        return load_pdf_file(file_path)

    return []
