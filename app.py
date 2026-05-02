"""
Semantic Search & RAG — REST API
==================================
Flask-based RESTful API providing endpoints for:
  - Document upload and management
  - Semantic search
  - RAG (retrieval-augmented generation)
  - Index statistics and health checks

Endpoints:
  POST   /api/documents          Upload a document
  GET    /api/documents          List all documents
  GET    /api/documents/<id>     Get document details
  DELETE /api/documents/<id>     Delete a document
  POST   /api/search             Semantic search
  POST   /api/rag                RAG query (search + generate)
  GET    /api/stats              Index statistics
  GET    /api/health             Health check
"""

import os
import time
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from vector_store import VectorStore
from rag import RAGPipeline, MockLLMClient, AnthropicClient, OpenAIClient


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------

def create_app(config: dict = None) -> Flask:
    """
    Application factory pattern for testability and configurability.
    """
    app = Flask(__name__)

    # Configuration with sensible defaults
    app.config.update({
        "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL", "tfidf"),
        "HNSW_M": int(os.environ.get("HNSW_M", "16")),
        "HNSW_EF_CONSTRUCTION": int(os.environ.get("HNSW_EF_CONSTRUCTION", "200")),
        "HNSW_EF_SEARCH": int(os.environ.get("HNSW_EF_SEARCH", "50")),
        "CHUNK_MAX_TOKENS": int(os.environ.get("CHUNK_MAX_TOKENS", "512")),
        "CHUNK_MIN_TOKENS": int(os.environ.get("CHUNK_MIN_TOKENS", "50")),
        "PERSIST_DIR": os.environ.get("PERSIST_DIR", None),
        "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "mock"),
        "MAX_UPLOAD_SIZE_MB": int(os.environ.get("MAX_UPLOAD_SIZE_MB", "10")),
    })

    if config:
        app.config.update(config)

    # Initialize the vector store
    store = VectorStore(
        embedding_model=app.config["EMBEDDING_MODEL"],
        hnsw_M=app.config["HNSW_M"],
        hnsw_ef_construction=app.config["HNSW_EF_CONSTRUCTION"],
        hnsw_ef_search=app.config["HNSW_EF_SEARCH"],
        chunk_max_tokens=app.config["CHUNK_MAX_TOKENS"],
        chunk_min_tokens=app.config["CHUNK_MIN_TOKENS"],
        persist_dir=app.config["PERSIST_DIR"],
    )

    # Load persisted data if available
    if app.config["PERSIST_DIR"]:
        try:
            store.load()
            app.logger.info("Loaded persisted index data.")
        except Exception:
            app.logger.info("No persisted data found, starting fresh.")

    # Initialize LLM client
    provider = app.config["LLM_PROVIDER"]
    if provider == "anthropic":
        llm = AnthropicClient()
    elif provider == "openai":
        llm = OpenAIClient()
    else:
        llm = MockLLMClient()

    # Initialize RAG pipeline
    rag = RAGPipeline(vector_store=store, llm_client=llm)

    # Store references on app for access in routes
    app.store = store
    app.rag = rag

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "message": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "message": str(e)}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "embedding_model": app.config["EMBEDDING_MODEL"],
            "llm_provider": app.config["LLM_PROVIDER"],
        })

    # --- Document Management ---

    @app.route("/api/documents", methods=["POST"])
    def upload_document():
        """
        Upload a document for indexing.

        Accepts either:
          - JSON body: {"content": "...", "title": "...", "metadata": {...}}
          - File upload: multipart form with 'file' field + optional 'title'

        Returns the document metadata including generated chunk IDs.
        """
        try:
            start = time.time()

            if request.is_json:
                data = request.get_json()
                content = data.get("content", "")
                title = data.get("title", "Untitled")
                metadata = data.get("metadata", {})
            elif "file" in request.files:
                file = request.files["file"]
                filename = secure_filename(file.filename or "upload.txt")
                content = file.read().decode("utf-8", errors="replace")
                title = request.form.get("title", filename)
                metadata = {"filename": filename}

                # Size check
                max_bytes = app.config["MAX_UPLOAD_SIZE_MB"] * 1024 * 1024
                if len(content.encode()) > max_bytes:
                    return jsonify({
                        "error": f"File exceeds {app.config['MAX_UPLOAD_SIZE_MB']}MB limit"
                    }), 413
            else:
                return jsonify({"error": "No content or file provided"}), 400

            if not content.strip():
                return jsonify({"error": "Document content is empty"}), 400

            # Ingest the document
            doc = store.add_document(content=content, title=title, metadata=metadata)

            elapsed = time.time() - start

            return jsonify({
                "id": doc.id,
                "title": doc.title,
                "num_chunks": len(doc.chunk_ids),
                "chunk_ids": doc.chunk_ids,
                "content_length": len(content),
                "processing_time_sec": round(elapsed, 3),
                "message": "Document indexed successfully",
            }), 201

        except Exception as e:
            app.logger.error(f"Upload error: {traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/documents", methods=["GET"])
    def list_documents():
        """List all indexed documents."""
        docs = []
        for doc in store.documents.values():
            docs.append({
                "id": doc.id,
                "title": doc.title,
                "num_chunks": len(doc.chunk_ids),
                "created_at": doc.created_at,
                "metadata": doc.metadata,
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            })

        docs.sort(key=lambda d: d["created_at"], reverse=True)
        return jsonify({"documents": docs, "total": len(docs)})

    @app.route("/api/documents/<doc_id>", methods=["GET"])
    def get_document(doc_id):
        """Get full document details including chunks."""
        if doc_id not in store.documents:
            return jsonify({"error": "Document not found"}), 404

        doc = store.documents[doc_id]
        chunks = [
            {
                "id": cid,
                "text": store.chunks[cid].text if cid in store.chunks else None,
                "index": store.chunks[cid].index if cid in store.chunks else None,
            }
            for cid in doc.chunk_ids
        ]

        return jsonify({
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "metadata": doc.metadata,
            "created_at": doc.created_at,
            "chunks": chunks,
        })

    @app.route("/api/documents/<doc_id>", methods=["DELETE"])
    def delete_document(doc_id):
        """Delete a document and its chunks from the index."""
        try:
            store.delete_document(doc_id)
            return jsonify({"message": f"Document {doc_id} deleted"})
        except KeyError:
            return jsonify({"error": "Document not found"}), 404

    # --- Search ---

    @app.route("/api/search", methods=["POST"])
    def search():
        """
        Semantic search across indexed documents.

        JSON body:
          {
            "query": "your search query",
            "k": 5,                        // optional, default 5
            "doc_id": "filter-doc-id"      // optional, filter to one doc
          }
        """
        data = request.get_json()
        if not data or not data.get("query"):
            return jsonify({"error": "Missing 'query' field"}), 400

        query = data["query"]
        k = data.get("k", 5)
        doc_filter = data.get("doc_id")

        start = time.time()
        results = store.search(query, k=k, doc_filter=doc_filter)
        elapsed = time.time() - start

        return jsonify({
            "query": query,
            "results": [r.to_dict() for r in results],
            "num_results": len(results),
            "search_time_sec": round(elapsed, 4),
        })

    # --- RAG ---

    @app.route("/api/rag", methods=["POST"])
    def rag_query():
        """
        RAG query: retrieves relevant context and generates an answer.

        JSON body:
          {
            "query": "your question",
            "k": 5                         // optional, chunks to retrieve
          }
        """
        data = request.get_json()
        if not data or not data.get("query"):
            return jsonify({"error": "Missing 'query' field"}), 400

        query = data["query"]
        k = data.get("k", 5)

        start = time.time()
        response = rag.query(query, k=k)
        elapsed = time.time() - start

        result = response.to_dict()
        result["response_time_sec"] = round(elapsed, 4)

        return jsonify(result)

    # --- Stats ---

    @app.route("/api/stats", methods=["GET"])
    def stats():
        """Return index and store statistics."""
        return jsonify(store.stats())

    # --- Persistence ---

    @app.route("/api/save", methods=["POST"])
    def save_index():
        """Persist the current index to disk."""
        if not app.config["PERSIST_DIR"]:
            return jsonify({"error": "Persistence not configured"}), 400
        try:
            store.save()
            return jsonify({"message": "Index saved successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"""
╔══════════════════════════════════════════════════════╗
║   Semantic Search & RAG Service                      ║
║   Running on http://localhost:{port}                   ║
║                                                      ║
║   Endpoints:                                         ║
║     POST   /api/documents     Upload document        ║
║     GET    /api/documents     List documents         ║
║     POST   /api/search        Semantic search        ║
║     POST   /api/rag           RAG query              ║
║     GET    /api/stats         Index stats            ║
║     GET    /api/health        Health check           ║
╚══════════════════════════════════════════════════════╝
    """)

    app.run(host="0.0.0.0", port=port, debug=debug)
