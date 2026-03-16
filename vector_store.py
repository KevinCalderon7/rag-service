"""
Vector Store
==============
Orchestration layer that ties together:
  - Document ingestion & storage
  - Semantic chunking
  - Embedding generation
  - HNSW index for vector retrieval

Provides a clean interface for the API layer to upload documents,
run semantic searches, and retrieve context for RAG.
"""

import json
import os
import time
import uuid
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional

from hnsw import HNSWIndex
from chunker import SemanticChunker, FixedSizeChunker, Chunk
from embeddings import EmbeddingService, TFIDFEmbeddingService


@dataclass
class Document:
    """A stored document with metadata."""
    id: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    score: float  # similarity score (higher = more relevant)
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class VectorStore:
    """
    Full-featured vector store for semantic search and RAG.

    Parameters
    ----------
    embedding_model : str
        SentenceTransformers model name or 'tfidf' for lightweight fallback.
    hnsw_M : int
        HNSW max connections per layer.
    hnsw_ef_construction : int
        HNSW build-time candidate list size.
    hnsw_ef_search : int
        HNSW query-time candidate list size.
    chunk_max_tokens : int
        Max tokens per chunk.
    chunk_min_tokens : int
        Min tokens per chunk.
    persist_dir : str, optional
        Directory for persisting index and documents to disk.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        hnsw_M: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
        chunk_max_tokens: int = 512,
        chunk_min_tokens: int = 50,
        persist_dir: Optional[str] = None,
    ):
        # Initialize embedding service
        if embedding_model == "tfidf":
            self.embedder = TFIDFEmbeddingService(dim=384)
            self._use_tfidf = True
        else:
            self.embedder = EmbeddingService(model_name=embedding_model)
            self._use_tfidf = False

        self._dim = 384  # default, updated on first embed

        # Chunker (initialized after embedder is ready)
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_min_tokens = chunk_min_tokens
        self._chunker = None

        # HNSW index (initialized lazily once we know the dim)
        self._hnsw_params = {
            "M": hnsw_M,
            "ef_construction": hnsw_ef_construction,
            "ef_search": hnsw_ef_search,
        }
        self._index: Optional[HNSWIndex] = None

        # Document & chunk storage
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}

        # Persistence
        self.persist_dir = persist_dir
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_index(self):
        """Create the HNSW index once we know the embedding dimension."""
        if self._index is None:
            if not self._use_tfidf:
                self._dim = self.embedder.dim
            self._index = HNSWIndex(
                dim=self._dim,
                metric="cosine",
                **self._hnsw_params,
            )

    def _ensure_chunker(self):
        """Create the semantic chunker with the embedding function."""
        if self._chunker is None:
            self._chunker = SemanticChunker(
                embed_fn=self.embedder,
                max_chunk_tokens=self.chunk_max_tokens,
                min_chunk_tokens=self.chunk_min_tokens,
            )

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_document(
        self,
        content: str,
        title: str = "Untitled",
        doc_id: Optional[str] = None,
        metadata: dict = None,
    ) -> Document:
        """
        Ingest a document: chunk it, embed chunks, and index them.

        Returns the Document object with chunk IDs populated.
        """
        self._ensure_index()
        self._ensure_chunker()

        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}

        # Semantic chunking
        chunks = self._chunker.chunk(content, doc_id, metadata)

        if not chunks:
            raise ValueError("Document produced no chunks. Content may be empty.")

        # If using TF-IDF, fit on this corpus (incremental)
        if self._use_tfidf:
            all_texts = [c.text for c in self.chunks.values()] + [c.text for c in chunks]
            self.embedder.fit(all_texts)

        # Embed all chunks in a batch
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(chunk_texts)

        # Store and index each chunk
        chunk_ids = []
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks[chunk.id] = chunk
            self._index.insert(chunk.id, embedding)
            chunk_ids.append(chunk.id)

        # Store the document
        doc = Document(
            id=doc_id,
            title=title,
            content=content,
            metadata=metadata,
            chunk_ids=chunk_ids,
        )
        self.documents[doc_id] = doc

        return doc

    def delete_document(self, doc_id: str):
        """Remove a document and all its chunks from the store."""
        if doc_id not in self.documents:
            raise KeyError(f"Document {doc_id} not found")

        doc = self.documents[doc_id]

        for chunk_id in doc.chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
            self._index.delete(chunk_id)

        del self.documents[doc_id]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        doc_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search across all indexed documents.

        Parameters
        ----------
        query : str
            Natural language search query.
        k : int
            Number of results to return.
        doc_filter : str, optional
            If provided, only return results from this document ID.

        Returns
        -------
        List of SearchResult objects sorted by relevance (highest first).
        """
        self._ensure_index()

        if len(self._index) == 0:
            return []

        # Embed the query
        query_vector = self.embedder.embed_query(query)

        # Search the HNSW index (request more results if filtering)
        search_k = k * 3 if doc_filter else k
        raw_results = self._index.query(query_vector, k=search_k)

        # Build SearchResult objects
        results = []
        for chunk_id, distance in raw_results:
            if chunk_id not in self.chunks:
                continue

            chunk = self.chunks[chunk_id]

            if doc_filter and chunk.doc_id != doc_filter:
                continue

            doc = self.documents.get(chunk.doc_id)
            doc_title = doc.title if doc else "Unknown"

            # Convert cosine distance to similarity score
            score = 1.0 - distance

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=doc_title,
                    text=chunk.text,
                    score=round(score, 4),
                    metadata=chunk.metadata,
                )
            )

        # Sort by score descending, take top k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    # ------------------------------------------------------------------
    # RAG context assembly
    # ------------------------------------------------------------------

    def get_rag_context(
        self,
        query: str,
        k: int = 5,
        max_context_tokens: int = 2000,
    ) -> dict:
        """
        Retrieve context for Retrieval Augmented Generation.

        Returns a dict with:
          - context: formatted context string for an LLM prompt
          - sources: list of source references
          - results: raw SearchResult objects
        """
        results = self.search(query, k=k)

        context_parts = []
        sources = []
        total_tokens = 0

        for i, result in enumerate(results):
            # Rough token estimate
            chunk_tokens = len(result.text) // 4
            if total_tokens + chunk_tokens > max_context_tokens:
                break

            context_parts.append(
                f"[Source {i + 1}: {result.doc_title} "
                f"(relevance: {result.score:.2f})]\n{result.text}"
            )
            sources.append({
                "index": i + 1,
                "doc_id": result.doc_id,
                "doc_title": result.doc_title,
                "chunk_id": result.chunk_id,
                "score": result.score,
            })
            total_tokens += chunk_tokens

        context = "\n\n---\n\n".join(context_parts)

        return {
            "context": context,
            "sources": sources,
            "results": results,
            "num_chunks_used": len(context_parts),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Persist documents and chunks to disk (JSON)."""
        if not self.persist_dir:
            raise ValueError("No persist_dir configured")

        # Save documents
        docs_data = {}
        for doc_id, doc in self.documents.items():
            docs_data[doc_id] = {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "metadata": doc.metadata,
                "created_at": doc.created_at,
                "chunk_ids": doc.chunk_ids,
            }

        with open(os.path.join(self.persist_dir, "documents.json"), "w") as f:
            json.dump(docs_data, f)

        # Save chunks
        chunks_data = {}
        for chunk_id, chunk in self.chunks.items():
            chunks_data[chunk_id] = {
                "id": chunk.id,
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata,
            }

        with open(os.path.join(self.persist_dir, "chunks.json"), "w") as f:
            json.dump(chunks_data, f)

        # Save embeddings (reconstruct by re-embedding — or save vectors)
        vectors = {}
        for chunk_id in self.chunks:
            if chunk_id in self._index.nodes:
                vectors[chunk_id] = self._index.nodes[chunk_id].vector.tolist()

        with open(os.path.join(self.persist_dir, "vectors.json"), "w") as f:
            json.dump(vectors, f)

    def load(self):
        """Load persisted documents, chunks, and vectors from disk."""
        if not self.persist_dir:
            raise ValueError("No persist_dir configured")

        docs_path = os.path.join(self.persist_dir, "documents.json")
        chunks_path = os.path.join(self.persist_dir, "chunks.json")
        vectors_path = os.path.join(self.persist_dir, "vectors.json")

        if not all(os.path.exists(p) for p in [docs_path, chunks_path, vectors_path]):
            return  # Nothing to load

        with open(docs_path) as f:
            docs_data = json.load(f)

        with open(chunks_path) as f:
            chunks_data = json.load(f)

        with open(vectors_path) as f:
            vectors = json.load(f)

        self._ensure_index()

        # Restore documents
        for doc_id, data in docs_data.items():
            self.documents[doc_id] = Document(**data)

        # Restore chunks and index vectors
        for chunk_id, data in chunks_data.items():
            self.chunks[chunk_id] = Chunk(**data)
            if chunk_id in vectors:
                vec = np.array(vectors[chunk_id], dtype=np.float32)
                self._index.insert(chunk_id, vec)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return store statistics."""
        self._ensure_index()
        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "index_stats": self._index.stats(),
        }
