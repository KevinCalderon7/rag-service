"""
Embedding Service
==================
Handles vector embedding generation for text chunks and queries.

Supports two backends:
  1. SentenceTransformers (recommended) — high-quality dense embeddings
  2. TF-IDF fallback — lightweight, no GPU needed, surprisingly decent for
     domain-specific corpora where the vocabulary is well-represented.

The service normalizes all embeddings to unit vectors for cosine similarity.
"""

import hashlib
import numpy as np
from typing import Optional
from functools import lru_cache


class EmbeddingService:
    """
    Generates text embeddings using sentence-transformers.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Good options:
        - 'Qwen 3.5'
    device : str
        'cpu', 'cuda', or 'auto'
    normalize : bool
        Whether to L2-normalize embeddings (recommended for cosine similarity).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.

        Returns
        -------
        np.ndarray of shape (len(texts), dim)
            Normalized embedding vectors.
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
        return self.embed([query])[0]

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Callable interface for use with SemanticChunker."""
        return self.embed(texts)


class TFIDFEmbeddingService:
    """
    Lightweight TF-IDF embedding fallback.

    Builds a vocabulary from the corpus and projects texts into a
    fixed-dimensional space using hashed TF-IDF features. No external
    model downloads required.

    Parameters
    ----------
    dim : int
        Output embedding dimension (uses feature hashing).
    """

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._idf: Optional[dict] = None
        self._vocab_size = 0

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens

    @staticmethod
    def _hash_token(token: str, dim: int) -> int:
        """Hash a token to a dimension index."""
        h = hashlib.md5(token.encode()).hexdigest()
        return int(h, 16) % dim

    def fit(self, corpus: list[str]):
        """Build IDF weights from a corpus."""
        doc_freq = {}
        n = len(corpus)

        for text in corpus:
            seen = set()
            for token in self._tokenize(text):
                if token not in seen:
                    doc_freq[token] = doc_freq.get(token, 0) + 1
                    seen.add(token)

        self._idf = {
            token: np.log((n + 1) / (df + 1)) + 1
            for token, df in doc_freq.items()
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate TF-IDF feature-hashed embeddings."""
        embeddings = np.zeros((len(texts), self._dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue

            # Term frequency
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            # TF-IDF with feature hashing
            for token, count in tf.items():
                idf = self._idf.get(token, 1.0) if self._idf else 1.0
                idx = self._hash_token(token, self._dim)
                embeddings[i, idx] += (count / len(tokens)) * idf

            # L2 normalize
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)
