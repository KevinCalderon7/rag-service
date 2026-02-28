"""
Semantic Document Chunker
==========================
Splits documents into chunks based on semantic boundaries rather than
arbitrary character counts. This preserves context and meaning within
each chunk.

Strategy (Hierarchical Semantic Splitting):
1. Split text into sentences.
2. Compute sentence embeddings.
3. Measure cosine similarity between consecutive sentence groups.
4. Detect "breakpoints" where similarity drops sharply — these are
   natural topic transitions.
5. Merge adjacent sentences into chunks at those breakpoints.
6. Apply min/max size constraints with overlap for continuity.
"""

import re
import numpy as np
from dataclasses import dataclass


@dataclass
class Chunk:
    """A semantically coherent text chunk with metadata."""
    id: str
    text: str
    doc_id: str
    index: int                    # position in the original document
    start_char: int               # character offset in original text
    end_char: int
    metadata: dict = None

    def __post_init__(self):
        self.metadata = self.metadata or {}


class SemanticChunker:
    """
    Chunks documents by detecting semantic boundaries between sentences.

    Parameters
    ----------
    embed_fn : callable
        Function that takes a list of strings and returns a numpy array
        of shape (n, dim) with their embeddings.
    max_chunk_tokens : int
        Soft upper bound on chunk size (in approximate tokens).
    min_chunk_tokens : int
        Minimum chunk size. Chunks below this are merged with neighbors.
    similarity_threshold : float
        Percentile breakpoint sensitivity. Lower = more splits.
        0.5 means split at points below the median similarity.
    window_size : int
        Number of sentences to group when computing inter-group similarity.
        Larger windows smooth out noise but may miss fine-grained boundaries.
    overlap_sentences : int
        Number of sentences to overlap between consecutive chunks
        for retrieval continuity.
    """

    def __init__(
        self,
        embed_fn: callable,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 50,
        similarity_threshold: float = 0.40,
        window_size: int = 3,
        overlap_sentences: int = 1,
    ):
        self.embed_fn = embed_fn
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.overlap_sentences = overlap_sentences

    # ------------------------------------------------------------------
    # Sentence segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[dict]:
        """
        Split text into sentences while preserving character offsets.
        Uses regex heuristics that handle common abbreviations and edge cases.
        """
        # Pattern: split on sentence-ending punctuation followed by whitespace
        # and a capital letter, but avoid splitting on common abbreviations.
        abbreviations = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bProf)(?<!\bSt)(?<!\bvs)(?<!\betc)(?<!\bInc)(?<!\bCo)(?<!\bJr)(?<!\bSr)"
        pattern = rf'{abbreviations}(?<=[.!?])\s+(?=[A-Z"\'(])'

        sentences = []
        last_end = 0

        for match in re.finditer(pattern, text):
            start = match.start()
            sent_text = text[last_end:start + 1].strip()
            if sent_text:
                sentences.append({
                    "text": sent_text,
                    "start": last_end,
                    "end": start + 1,
                })
            last_end = match.end()

        # Capture the final sentence
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append({
                "text": remaining,
                "start": last_end,
                "end": len(text),
            })

        return sentences

    # ------------------------------------------------------------------
    # Semantic similarity computation
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _compute_breakpoints(
        self, sentences: list[dict]
    ) -> list[int]:
        """
        Find semantic breakpoints between sentence groups.

        1. Group sentences into overlapping windows.
        2. Embed each window (by concatenating its sentences).
        3. Compute cosine similarity between consecutive windows.
        4. Mark positions where similarity drops below a dynamic threshold.

        Returns list of sentence indices where splits should occur.
        """
        n = len(sentences)
        if n <= self.window_size:
            return []

        # Build windows of sentences
        windows = []
        for i in range(n - self.window_size + 1):
            window_text = " ".join(
                s["text"] for s in sentences[i : i + self.window_size]
            )
            windows.append(window_text)

        # Embed all windows in a single batch
        embeddings = self.embed_fn(windows)

        # Compute similarities between consecutive windows
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        if not similarities:
            return []

        # Dynamic threshold: use percentile-based cutoff
        # Points below this threshold are considered breakpoints
        threshold = np.percentile(similarities, self.similarity_threshold * 100)

        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                # The breakpoint is at the end of the window
                # i.e., between sentence (i + window_size - 1) and the next
                bp = i + self.window_size
                if bp < n:
                    breakpoints.append(bp)

        return sorted(set(breakpoints))

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate (~4 chars per token for English)."""
        return len(text) // 4

    # ------------------------------------------------------------------
    # Main chunking pipeline
    # ------------------------------------------------------------------

    def chunk(self, text: str, doc_id: str, metadata: dict = None) -> list[Chunk]:
        """
        Split a document into semantically coherent chunks.

        Pipeline:
        1. Sentence segmentation
        2. Semantic breakpoint detection
        3. Group sentences into chunks at breakpoints
        4. Enforce min/max size constraints
        5. Add overlap between consecutive chunks

        Returns list of Chunk objects.
        """
        metadata = metadata or {}
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # For very short documents, return as single chunk
        if len(sentences) <= 3:
            return [
                Chunk(
                    id=f"{doc_id}_chunk_0",
                    text=text.strip(),
                    doc_id=doc_id,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata=metadata,
                )
            ]

        # Step 1: Find semantic breakpoints
        breakpoints = self._compute_breakpoints(sentences)

        # Step 2: Group sentences into initial segments
        segments = []
        prev = 0
        for bp in breakpoints:
            segments.append(sentences[prev:bp])
            prev = bp
        segments.append(sentences[prev:])

        # Step 3: Enforce max size — split oversized segments
        refined = []
        for seg in segments:
            seg_text = " ".join(s["text"] for s in seg)
            if self._estimate_tokens(seg_text) > self.max_chunk_tokens:
                # Sub-split evenly
                mid = len(seg) // 2
                refined.append(seg[:mid])
                refined.append(seg[mid:])
            else:
                refined.append(seg)

        # Step 4: Enforce min size — merge undersized segments
        merged = []
        buffer = []
        for seg in refined:
            buffer.extend(seg)
            buf_text = " ".join(s["text"] for s in buffer)
            if self._estimate_tokens(buf_text) >= self.min_chunk_tokens:
                merged.append(buffer)
                buffer = []
        if buffer:
            if merged:
                merged[-1].extend(buffer)
            else:
                merged.append(buffer)

        # Step 5: Build final chunks with overlap
        chunks = []
        for i, seg in enumerate(merged):
            # Add overlap from previous segment
            if i > 0 and self.overlap_sentences > 0:
                prev_seg = merged[i - 1]
                overlap = prev_seg[-self.overlap_sentences:]
                full_seg = overlap + seg
            else:
                full_seg = seg

            chunk_text = " ".join(s["text"] for s in full_seg)
            start_char = full_seg[0]["start"]
            end_char = full_seg[-1]["end"]

            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{i}",
                    text=chunk_text,
                    doc_id=doc_id,
                    index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={**metadata, "num_sentences": len(full_seg)},
                )
            )

        return chunks


class FixedSizeChunker:
    """
    Fallback chunker that splits by token count with overlap.
    Useful when embedding function is unavailable or for simple use cases.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str, metadata: dict = None) -> list[Chunk]:
        metadata = metadata or {}
        words = text.split()
        chunks = []
        start = 0
        idx = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Approximate char offsets
            start_char = len(" ".join(words[:start]))
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{idx}",
                    text=chunk_text,
                    doc_id=doc_id,
                    index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata,
                )
            )

            start += self.chunk_size - self.overlap
            idx += 1

        return chunks
