"""
chunking.py
------------
Splits extracted document text into smaller overlapping chunks
using LangChain's RecursiveCharacterTextSplitter, while preserving
source metadata (filename, page number) for each chunk.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def chunk_documents(
    extracted_data: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[dict]:
    """
    Takes the output of document_processor.process_multiple_documents()
    i.e. a list of {"text": ..., "source": ..., "page": ...} dicts,
    and splits each entry's text into smaller chunks.

    Returns a list of dicts:
        {
            "chunk_text": ...,
            "source": ...,
            "page": ...,
            "chunk_id": ...
        }
    """
    if not extracted_data:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    chunk_counter = 0

    for entry in extracted_data:
        text = entry.get("text", "")
        source = entry.get("source", "unknown")
        page = entry.get("page", 1)

        if not text.strip():
            continue

        split_texts = splitter.split_text(text)

        for split_text in split_texts:
            chunk_counter += 1
            all_chunks.append({
                "chunk_text": split_text,
                "source": source,
                "page": page,
                "chunk_id": chunk_counter
            })

    return all_chunks