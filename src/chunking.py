from typing import Dict, List

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE
)


def clean_text(text: str) -> str:

    return " ".join(
        text.split()
    )


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end]

        if end < text_length:

            last_period = chunk.rfind(". ")

            if last_period > chunk_size // 2:

                end = (
                    start
                    + last_period
                    + 1
                )

                chunk = text[start:end]

        chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(
            end - chunk_overlap,
            start + 1
        )

    return chunks


def chunk_documents(
    documents: List[Dict]
) -> List[Dict]:

    chunked_documents = []

    chunk_counter = 0

    for document in documents:

        chunks = split_text(
            document["text"]
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            metadata = document[
                "metadata"
            ].copy()

            metadata["chunk_index"] = (
                chunk_index
            )

            chunked_documents.append(
                {
                    "id": f"chunk-{chunk_counter}",
                    "text": chunk,
                    "metadata": metadata
                }
            )

            chunk_counter += 1

    return chunked_documents
