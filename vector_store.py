"""
vector_store.py
----------------
Generates embeddings for document chunks using Sentence Transformers
(all-MiniLM-L6-v2) and stores/searches them using a FAISS vector database.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    """
    Wraps a FAISS index + the raw chunk metadata so we can:
    - build/rebuild the index from a list of chunks
    - run similarity search and get back the original chunk text + source info
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None
        self.chunks = []  # keeps chunk_text/source/page/chunk_id aligned with FAISS vector order

    def build_index(self, chunks: list[dict]):
        """
        Takes the output of chunking.chunk_documents():
            [{"chunk_text": ..., "source": ..., "page": ..., "chunk_id": ...}, ...]
        Embeds all chunk texts and builds a new FAISS index.
        """
        if not chunks:
            raise ValueError("No chunks provided to build the vector index.")

        self.chunks = chunks
        texts = [c["chunk_text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # so we can use inner product as cosine similarity
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # inner product = cosine sim (since normalized)
        self.index.add(embeddings.astype(np.float32))

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embeds the query and retrieves the top_k most similar chunks.
        Returns a list of dicts (same shape as input chunks) with an added "score" field.
        """
        if not self.is_ready():
            raise ValueError("Vector index is empty. Please upload and process documents first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        top_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])  # copy to avoid mutating original
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def reset(self):
        """Clear the index (e.g. when user uploads a new set of documents)."""
        self.index = None
        self.chunks = []