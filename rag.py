"""
RAG (Retrieval-Augmented Generation) Pipeline
===============================================
Combines the vector store's semantic search with LLM generation
to answer questions grounded in the indexed documents.

Supports multiple LLM backends:
  - Anthropic Claude API
  - OpenAI API
  - Local/custom endpoints

The pipeline handles:
  1. Query analysis & rewriting
  2. Context retrieval from the vector store
  3. Prompt construction with source attribution
  4. LLM generation with grounding
"""

import os
import json
from typing import Optional
from dataclasses import dataclass, field, asdict

from vector_store import VectorStore


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions \
based on the provided context. Follow these rules:

1. Answer ONLY based on the provided context. If the context doesn't contain \
enough information, say so clearly.
2. Cite your sources using [Source N] notation when making claims.
3. Be concise and direct.
4. If the question is ambiguous, ask for clarification.
5. Never fabricate information not present in the context."""

RAG_USER_TEMPLATE = """Context from indexed documents:
---
{context}
---

Question: {query}

Provide a clear, well-sourced answer based on the context above."""


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""
    answer: str
    sources: list[dict]
    query: str
    model: str
    num_chunks_used: int
    context_preview: str = ""  # first 200 chars of context for debugging

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------
# LLM Client abstraction
# ------------------------------------------------------------------

class LLMClient:
    """
    Abstract interface for LLM generation.
    Subclass this to add new LLM backends.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Claude API client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content


class MockLLMClient(LLMClient):
    """
    Mock client for testing — echoes the context back.
    Also useful when you want retrieval without generation costs.
    """

    def __init__(self):
        self.model = "mock"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "**[Mock LLM — showing retrieved context]**\n\n"
            f"{user_prompt}\n\n"
            "_In production, this would be answered by an LLM using the above context._"
        )


# ------------------------------------------------------------------
# RAG Pipeline
# ------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Parameters
    ----------
    vector_store : VectorStore
        The indexed document store for retrieval.
    llm_client : LLMClient
        The LLM backend for generation.
    k : int
        Number of chunks to retrieve per query.
    max_context_tokens : int
        Maximum context window to send to the LLM.
    system_prompt : str
        System prompt for the LLM.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: Optional[LLMClient] = None,
        k: int = 5,
        max_context_tokens: int = 2000,
        system_prompt: str = RAG_SYSTEM_PROMPT,
    ):
        self.store = vector_store
        self.llm = llm_client or MockLLMClient()
        self.k = k
        self.max_context_tokens = max_context_tokens
        self.system_prompt = system_prompt

    def query(self, question: str, k: Optional[int] = None) -> RAGResponse:
        """
        Run the full RAG pipeline: retrieve → construct prompt → generate.

        Parameters
        ----------
        question : str
            The user's natural language question.
        k : int, optional
            Override the default number of chunks to retrieve.

        Returns
        -------
        RAGResponse with the answer, sources, and metadata.
        """
        k = k or self.k

        # Step 1: Retrieve relevant context
        rag_context = self.store.get_rag_context(
            query=question,
            k=k,
            max_context_tokens=self.max_context_tokens,
        )

        context = rag_context["context"]
        sources = rag_context["sources"]

        if not context:
            return RAGResponse(
                answer="I couldn't find any relevant information in the indexed documents to answer your question.",
                sources=[],
                query=question,
                model=getattr(self.llm, "model", "unknown"),
                num_chunks_used=0,
            )

        # Step 2: Construct the prompt
        user_prompt = RAG_USER_TEMPLATE.format(
            context=context,
            query=question,
        )

        # Step 3: Generate the answer
        answer = self.llm.generate(self.system_prompt, user_prompt)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            model=getattr(self.llm, "model", "unknown"),
            num_chunks_used=rag_context["num_chunks_used"],
            context_preview=context[:200],
        )

    def search_only(self, query: str, k: Optional[int] = None) -> list[dict]:
        """
        Run retrieval without generation. Useful for debugging
        or when you just need ranked document passages.
        """
        k = k or self.k
        results = self.store.search(query, k=k)
        return [r.to_dict() for r in results]
