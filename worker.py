"""
Async Document Ingestion Worker
=================================
For AWS deployment: handles document ingestion asynchronously via
a task queue. When a user uploads a document, the API returns immediately
with a job ID, and this worker processes the document in the background.

Backends:
  - Celery + Redis (default, simple)
  - Celery + AWS SQS (production, managed)

Usage:
  celery -A worker worker --loglevel=info --concurrency=2
"""

import os
import time
import json
from celery import Celery

# Configure Celery
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery("rag_worker", broker=BROKER_URL, backend=RESULT_BACKEND)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # Retry on worker crash
    worker_prefetch_multiplier=1,  # Fair scheduling
)


# Lazy-loaded shared vector store instance
_store = None


def get_store():
    """Get or create the shared vector store instance."""
    global _store
    if _store is None:
        from vector_store import VectorStore

        _store = VectorStore(
            embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            persist_dir=os.environ.get("PERSIST_DIR", "/data"),
        )
        try:
            _store.load()
        except Exception:
            pass
    return _store


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def ingest_document(self, content: str, title: str, metadata: dict = None):
    """
    Async task: ingest a document into the vector store.

    Parameters
    ----------
    content : str
        Document text content.
    title : str
        Document title.
    metadata : dict
        Optional metadata.

    Returns
    -------
    dict with document ID, chunk count, and processing time.
    """
    try:
        self.update_state(state="PROCESSING", meta={"title": title})

        store = get_store()
        start = time.time()

        doc = store.add_document(
            content=content,
            title=title,
            metadata=metadata or {},
        )

        # Persist after each ingestion
        try:
            store.save()
        except Exception:
            pass  # Non-fatal

        elapsed = time.time() - start

        return {
            "status": "completed",
            "doc_id": doc.id,
            "title": doc.title,
            "num_chunks": len(doc.chunk_ids),
            "processing_time_sec": round(elapsed, 3),
        }

    except Exception as exc:
        self.update_state(state="FAILED", meta={"error": str(exc)})
        raise self.retry(exc=exc)


@celery_app.task
def batch_ingest(documents: list[dict]):
    """
    Ingest multiple documents in sequence.

    Parameters
    ----------
    documents : list of dict
        Each dict has: content, title, metadata (optional).
    """
    results = []
    for doc_data in documents:
        result = ingest_document.delay(
            content=doc_data["content"],
            title=doc_data.get("title", "Untitled"),
            metadata=doc_data.get("metadata", {}),
        )
        results.append(result.id)
    return {"task_ids": results, "total": len(results)}
