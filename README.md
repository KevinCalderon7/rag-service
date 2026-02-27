# Semantic Search & RAG Service

A production-grade semantic search and Retrieval-Augmented Generation (RAG) service built from the ground up, featuring a **custom HNSW index** (no vector database dependency), **semantic document chunking**, and a **RESTful Flask API**.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Flask REST API                         │
│  POST /documents  POST /search  POST /rag  GET /stats       │
└──────────────┬───────────────────────────┬───────────────────┘
               │                           │
       ┌───────▼───────┐          ┌────────▼────────┐
       │  Vector Store  │          │  RAG Pipeline   │
       │  (Orchestrator)│◄────────►│  (Retrieval +   │
       │                │          │   Generation)   │
       └───┬───────┬────┘          └────────┬────────┘
           │       │                        │
   ┌───────▼──┐ ┌──▼──────────┐   ┌────────▼────────┐
   │  HNSW    │ │  Semantic   │   │  LLM Client     │
   │  Index   │ │  Chunker    │   │  (Ollama OOM)   │
   │ (custom) │ │             │   │                 │
   └──────────┘ └─────────────┘   └─────────────────┘
```

## Key Components

### 1. Custom HNSW Index (`hnsw.py`)
From-scratch implementation of the Hierarchical Navigable Small World algorithm:
- **Multi-layer graph** where higher layers act as express lanes for search
- **Heuristic neighbor selection** for diverse, navigable graph structure
- **Beam search** at layer 0 with configurable `ef_search` for recall/speed tradeoff
- Thread-safe insertions with locking
- Supports cosine and euclidean distance metrics

### 2. Semantic Chunker (`chunker.py`)
Intelligently splits documents at semantic boundaries:
- Embeds sliding windows of sentences and measures inter-window similarity
- Detects **topic transition points** where similarity drops below a dynamic threshold
- Enforces min/max chunk size constraints with sentence-level overlap
- Preserves character offsets for source attribution

### 3. Embedding Service (`embeddings.py`)
- **SentenceTransformers** backend (e.g., `all-MiniLM-L6-v2`) for production quality
- **TF-IDF fallback** with feature hashing — no model download, no GPU needed

### 4. RAG Pipeline (`rag.py`)
- Retrieves relevant chunks, assembles context with source attribution
- Pluggable LLM backends: Anthropic Claude, OpenAI, or local Ollama OOM
- Configurable context window and retrieval depth

## Quick Start

### Local Development (TF-IDF mode, no GPU needed)

```bash
pip install flask numpy pytest

# Start the server
python app.py

# Upload a document
curl -X POST http://localhost:5000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"title": "AI Overview", "content": "Machine learning is a subset of artificial intelligence..."}'

# Search
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is machine learning", "k": 3}'

# RAG query
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "explain machine learning"}'
```

### Production (SentenceTransformers + LLM)

```bash
pip install -r requirements.txt

# Configure
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export PERSIST_DIR=./data

python app.py
```

### Docker Compose (with async worker)

```bash
docker-compose up --build
```

This starts the API server, a Celery worker for async document ingestion, and Redis as the message broker.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/documents` | POST | Upload & index a document |
| `/api/documents` | GET | List all documents |
| `/api/documents/<id>` | GET | Get document details + chunks |
| `/api/documents/<id>` | DELETE | Remove document from index |
| `/api/search` | POST | Semantic search (`{"query": "...", "k": 5}`) |
| `/api/rag` | POST | RAG query (`{"query": "...", "k": 5}`) |
| `/api/stats` | GET | Index statistics |
| `/api/save` | POST | Persist index to disk |

## Running Tests

```bash
cd rag-service
pytest tests/ -v
```

The test suite verifies:
- HNSW recall@10 against brute-force (>70% on random data)
- Insertion and query throughput benchmarks
- Semantic chunker boundary detection
- Full API endpoint integration tests

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `tfidf` | `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, or `tfidf` |
| `LLM_PROVIDER` | `mock` | `anthropic`, `openai`, or `mock` |
| `HNSW_M` | `16` | Max connections per layer (higher = better recall) |
| `HNSW_EF_CONSTRUCTION` | `200` | Build-time candidate list (higher = better index) |
| `HNSW_EF_SEARCH` | `50` | Query-time candidate list (higher = better recall) |
| `CHUNK_MAX_TOKENS` | `512` | Max chunk size in approx tokens |
| `CHUNK_MIN_TOKENS` | `50` | Min chunk size in approx tokens |
| `PERSIST_DIR` | `None` | Directory for persisting index to disk |
| `PORT` | `5000` | Server port |

## AWS Deployment

For production on AWS:

1. **ECS Fargate** — Run the Docker container as a service
2. **SQS** — Replace Redis with SQS as the Celery broker (`CELERY_BROKER_URL=sqs://`)
3. **EFS** — Mount as the persist directory for durable index storage
4. **ALB** — Application Load Balancer in front of the API
5. **CloudWatch** — Logs and metrics from the containers

The `worker.py` module provides Celery tasks for asynchronous document ingestion, with automatic retries and state tracking.
