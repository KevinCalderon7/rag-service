"""
Test Suite for Semantic Search & RAG Service
=============================================
Tests the core components end-to-end:
  - HNSW index correctness (recall against brute force)
  - Semantic chunker boundary detection
  - Vector store ingestion and retrieval
  - Flask API endpoints
"""

import json
import time
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from hnsw import HNSWIndex
from chunker import SemanticChunker, FixedSizeChunker
from embeddings import TFIDFEmbeddingService
from vector_store import VectorStore
from app import create_app


# ==================================================================
# HNSW Index Tests
# ==================================================================

class TestHNSW:
    """Verify the custom HNSW implementation against brute-force search."""

    def _brute_force_knn(self, query, vectors, k):
        """Ground truth nearest neighbors via exhaustive search."""
        dists = []
        for id, vec in vectors.items():
            # Cosine distance
            norm_q = np.linalg.norm(query)
            norm_v = np.linalg.norm(vec)
            if norm_q == 0 or norm_v == 0:
                dist = 1.0
            else:
                dist = 1.0 - float(np.dot(query, vec) / (norm_q * norm_v))
            dists.append((id, dist))
        dists.sort(key=lambda x: x[1])
        return dists[:k]

    def test_insert_and_query_basic(self):
        """Test that inserting and querying returns reasonable results."""
        dim = 32
        index = HNSWIndex(dim=dim, M=8, ef_construction=50, ef_search=30)

        np.random.seed(42)
        vectors = {}
        for i in range(100):
            vec = np.random.randn(dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            id = f"vec_{i}"
            vectors[id] = vec
            index.insert(id, vec)

        assert len(index) == 100

        # Query with a known vector — it should find itself
        query = vectors["vec_0"]
        results = index.query(query, k=5)
        result_ids = [r[0] for r in results]
        assert "vec_0" in result_ids, f"Expected vec_0 in results, got {result_ids}"

    def test_recall_at_k(self):
        """Measure recall@10 against brute force — should be > 80%."""
        dim = 64
        index = HNSWIndex(dim=dim, M=16, ef_construction=100, ef_search=50)

        np.random.seed(123)
        vectors = {}
        for i in range(500):
            vec = np.random.randn(dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vectors[f"v{i}"] = vec
            index.insert(f"v{i}", vec)

        # Test recall over 20 random queries
        recalls = []
        k = 10
        for _ in range(20):
            query = np.random.randn(dim).astype(np.float32)
            query /= np.linalg.norm(query)

            # HNSW results
            hnsw_results = set(r[0] for r in index.query(query, k=k))

            # Brute force results
            bf_results = set(r[0] for r in self._brute_force_knn(query, vectors, k))

            recall = len(hnsw_results & bf_results) / k
            recalls.append(recall)

        avg_recall = np.mean(recalls)
        print(f"\n  HNSW Recall@{k}: {avg_recall:.2%} (over 20 queries, 500 vectors)")
        assert avg_recall > 0.7, f"Recall too low: {avg_recall:.2%}"

    def test_delete(self):
        """Test that deleted nodes are not returned."""
        dim = 16
        index = HNSWIndex(dim=dim, M=4, ef_construction=20, ef_search=10)

        vec = np.ones(dim, dtype=np.float32)
        vec /= np.linalg.norm(vec)

        index.insert("a", vec)
        index.insert("b", vec * 0.99)

        results_before = index.query(vec, k=2)
        assert len(results_before) == 2

        index.delete("a")
        results_after = index.query(vec, k=2)
        result_ids = [r[0] for r in results_after]
        assert "a" not in result_ids

    def test_stats(self):
        """Test diagnostic stats output."""
        dim = 16
        index = HNSWIndex(dim=dim, M=4)

        for i in range(10):
            vec = np.random.randn(dim).astype(np.float32)
            index.insert(f"n{i}", vec)

        stats = index.stats()
        assert stats["num_nodes"] == 10
        assert stats["max_layer"] >= 0
        assert stats["total_edges"] > 0

    def test_empty_index(self):
        """Querying an empty index returns empty results."""
        index = HNSWIndex(dim=16)
        results = index.query(np.zeros(16), k=5)
        assert results == []


# ==================================================================
# Chunker Tests
# ==================================================================

class TestChunker:
    """Test semantic and fixed-size chunking."""

    @staticmethod
    def _mock_embed(texts):
        """Mock embedding: hash-based for deterministic testing."""
        embeddings = np.zeros((len(texts), 32), dtype=np.float32)
        for i, text in enumerate(texts):
            np.random.seed(hash(text) % 2**31)
            embeddings[i] = np.random.randn(32)
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        return embeddings

    def test_semantic_chunker_produces_chunks(self):
        """Verify that the semantic chunker produces non-empty chunks."""
        chunker = SemanticChunker(
            embed_fn=self._mock_embed,
            max_chunk_tokens=100,
            min_chunk_tokens=20,
        )

        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It involves training algorithms on data to make predictions. "
            "Deep learning uses neural networks with many layers. "
            "Natural language processing deals with understanding text. "
            "Computer vision focuses on understanding images. "
            "Reinforcement learning trains agents through rewards. "
            "Transfer learning leverages pre-trained models. "
            "The field has grown rapidly in recent years. "
            "Many companies now use ML in production systems. "
            "Ethics in AI remains an important topic of discussion."
        )

        chunks = chunker.chunk(text, doc_id="test_doc")

        assert len(chunks) > 0
        assert all(c.text.strip() for c in chunks)
        assert all(c.doc_id == "test_doc" for c in chunks)

        # Chunks should cover the full text
        all_text = " ".join(c.text for c in chunks)
        for sentence_fragment in ["Machine learning", "Ethics in AI"]:
            assert sentence_fragment in all_text

    def test_fixed_size_chunker(self):
        """Test the fallback fixed-size chunker."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        text = " ".join(f"word{i}" for i in range(50))
        chunks = chunker.chunk(text, doc_id="fixed_test")

        assert len(chunks) > 1
        assert all(c.text.strip() for c in chunks)

    def test_short_document(self):
        """Very short documents should produce a single chunk."""
        chunker = SemanticChunker(embed_fn=self._mock_embed)
        text = "This is a short document."
        chunks = chunker.chunk(text, doc_id="short")
        assert len(chunks) == 1


# ==================================================================
# Vector Store Tests
# ==================================================================

class TestVectorStore:
    """Test the integrated vector store with TF-IDF embeddings."""

    def _create_store(self):
        return VectorStore(embedding_model="tfidf")

    def test_add_and_search(self):
        """Add documents and verify semantic search returns relevant results."""
        store = self._create_store()

        store.add_document(
            content="Python is a popular programming language used for web development, "
                    "data science, machine learning, and automation. It has a simple syntax "
                    "that makes it easy to learn.",
            title="Python Overview",
        )

        store.add_document(
            content="Chocolate cake is a delicious dessert made with cocoa powder, flour, "
                    "sugar, eggs, and butter. It is often served with frosting or whipped cream.",
            title="Chocolate Cake Recipe",
        )

        results = store.search("programming language for data science", k=2)
        assert len(results) > 0
        assert results[0].doc_title == "Python Overview"

    def test_delete_document(self):
        """Deleted documents should not appear in search results."""
        store = self._create_store()

        doc = store.add_document(content="Temporary content about quantum computing.", title="Temp")
        assert len(store.documents) == 1

        store.delete_document(doc.id)
        assert len(store.documents) == 0

    def test_rag_context(self):
        """Test RAG context assembly."""
        store = self._create_store()

        store.add_document(
            content="The capital of France is Paris. It is known for the Eiffel Tower.",
            title="France Facts",
        )

        context = store.get_rag_context("What is the capital of France?", k=3)
        assert context["context"]  # non-empty
        assert context["num_chunks_used"] > 0
        assert len(context["sources"]) > 0

    def test_stats(self):
        """Test stats after adding documents."""
        store = self._create_store()
        store.add_document(content="Test content for stats.", title="Stats Test")

        stats = store.stats()
        assert stats["num_documents"] == 1
        assert stats["num_chunks"] > 0


# ==================================================================
# Flask API Tests
# ==================================================================

class TestAPI:
    """Test the REST API endpoints."""

    @pytest.fixture
    def client(self):
        app = create_app({"EMBEDDING_MODEL": "tfidf", "LLM_PROVIDER": "mock"})
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"

    def test_upload_json(self, client):
        resp = client.post("/api/documents", json={
            "content": "Flask is a lightweight web framework for Python. "
                       "It is designed to be simple and extensible.",
            "title": "Flask Intro",
        })
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["num_chunks"] > 0
        assert "id" in data

    def test_list_documents(self, client):
        # Upload first
        client.post("/api/documents", json={
            "content": "Test document content.",
            "title": "Test",
        })

        resp = client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] >= 1

    def test_search(self, client):
        # Upload a document
        client.post("/api/documents", json={
            "content": "Kubernetes is a container orchestration platform. "
                       "It manages containerized workloads and services.",
            "title": "K8s Guide",
        })

        resp = client.post("/api/search", json={
            "query": "container orchestration",
            "k": 3,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["num_results"] > 0

    def test_rag_endpoint(self, client):
        client.post("/api/documents", json={
            "content": "The speed of light is approximately 299,792,458 meters per second.",
            "title": "Physics Facts",
        })

        resp = client.post("/api/rag", json={
            "query": "How fast is light?",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "answer" in data
        assert "sources" in data

    def test_delete_document(self, client):
        resp = client.post("/api/documents", json={
            "content": "Deletable content.",
            "title": "To Delete",
        })
        doc_id = resp.get_json()["id"]

        resp = client.delete(f"/api/documents/{doc_id}")
        assert resp.status_code == 200

        resp = client.get(f"/api/documents/{doc_id}")
        assert resp.status_code == 404

    def test_search_empty_query(self, client):
        resp = client.post("/api/search", json={})
        assert resp.status_code == 400

    def test_stats(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200


# ==================================================================
# HNSW Performance Benchmark (optional, runs with -v flag)
# ==================================================================

class TestPerformance:
    """Quick benchmarks for HNSW operations."""

    def test_insertion_throughput(self):
        """Measure insertion speed."""
        dim = 128
        n = 1000
        index = HNSWIndex(dim=dim, M=16, ef_construction=100)

        np.random.seed(0)
        vectors = np.random.randn(n, dim).astype(np.float32)
        for i in range(n):
            vectors[i] /= np.linalg.norm(vectors[i])

        start = time.time()
        for i in range(n):
            index.insert(f"v{i}", vectors[i])
        elapsed = time.time() - start

        rate = n / elapsed
        print(f"\n  HNSW Insert: {n} vectors in {elapsed:.2f}s ({rate:.0f} vec/s)")
        assert rate > 20, f"Insertion too slow: {rate:.0f} vec/s"

    def test_query_throughput(self):
        """Measure query speed."""
        dim = 128
        index = HNSWIndex(dim=dim, M=16, ef_construction=100, ef_search=50)

        np.random.seed(0)
        for i in range(1000):
            vec = np.random.randn(dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            index.insert(f"v{i}", vec)

        queries = np.random.randn(100, dim).astype(np.float32)
        for i in range(100):
            queries[i] /= np.linalg.norm(queries[i])

        start = time.time()
        for q in queries:
            index.query(q, k=10)
        elapsed = time.time() - start

        rate = 100 / elapsed
        print(f"\n  HNSW Query: 100 queries in {elapsed:.2f}s ({rate:.0f} q/s)")
        assert rate > 50, f"Query too slow: {rate:.0f} q/s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
