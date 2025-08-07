# rag_query_pipeline
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

# -----------------------------------------------------------------------------
# Environment / Globals
# -----------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("⚠️  OPENAI_API_KEY not found; set it in .env or environment")

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

# -----------------------------------------------------------------------------
# Load resources
# -----------------------------------------------------------------------------

def load_resources(index_path: str | Path,
                   reranker_model: str = "BAAI/bge-reranker-large") -> tuple[FAISS, OpenAIEmbeddings, CrossEncoder]:
    """Load FAISS store, embeddings, and reranker model."""
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,
        openai_api_key=OPENAI_API_KEY,
    )

    vectordb = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    reranker = CrossEncoder(reranker_model)
    return vectordb, embeddings, reranker

# -----------------------------------------------------------------------------
# Retrieval & rerank
# -----------------------------------------------------------------------------

def retrieve(query: str, vectordb: FAISS, embeddings: OpenAIEmbeddings, k1: int = 100) -> List[Document]:
    """Return top‑k1 docs via vector similarity search."""
    return vectordb.similarity_search(query, k=k1)


def rerank(query: str, docs: List[Document], reranker: CrossEncoder, k2: int = 10) -> List[Document]:
    """Rerank docs with cross‑encoder and return top‑k2."""
    if not docs:
        return []

    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)  # higher is more relevant
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:k2]]

# -----------------------------------------------------------------------------
# LLM answer generation
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an AI assistant that answers employee questions using our internal knowledge base."
    "Our company is a social-casino game studio"
    "The context contains Slack messages, file excerpts, or summaries retrieved for high relevance to the question"
    "If the answer is not contained in the context, say you don't know."
    "Give concrete numbers, dates, names, and policies when available."
    "Preserve original terminology; add concise explanations only if helpful."
    "Whenever the context contains garbled symbols like ‘�’, intelligently infer and replace the missing text from surrounding context, then reply with a clean, coherent answer free of any garbled characters."
)


def build_prompt(query: str, context_docs: List[Document]) -> List[dict]:
    """Create messages list for OpenAI chat completion."""
    context = "\n\n".join(f"[Doc {i+1}]\n" + d.page_content for i, d in enumerate(context_docs))
    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def generate_answer(query: str,
                    context_docs: List[Document],
                    model: str = "gpt-4o-mini",
                    max_tokens: int = 512) -> str:
    """Feed context + query to LLM and return its answer text."""
    chat = OpenAI(model=model, temperature=0.2, openai_api_key=OPENAI_API_KEY)
    messages = build_prompt(query, context_docs)
    response = chat.invoke(messages, max_tokens=max_tokens)  # returns str in langchain>=0.2
    return response

# -----------------------------------------------------------------------------
# High‑level helper
# -----------------------------------------------------------------------------

def answer_query(query: str,
                 index_path: str | Path,
                 k1: int = 100,
                 k2: int = 10,
                 reranker_model: str = "BAAI/bge-reranker-large",
                 llm_model: str = "gpt-4o-mini") -> Tuple[str, List[Document]]:
    """End‑to‑end convenience wrapper. Returns (answer, used_docs)."""
    vectordb, embeddings, reranker = load_resources(index_path, reranker_model)
    candidates = retrieve(query, vectordb, embeddings, k1=k1)
    top_docs = rerank(query, candidates, reranker, k2=k2)
    answer = generate_answer(query, top_docs, model=llm_model)
    return answer, top_docs

# -----------------------------------------------------------------------------
# CLI test harness (optional)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, textwrap

    p = argparse.ArgumentParser(description="Test the RAG query pipeline")
    p.add_argument("query", help="User question")
    p.add_argument("--index", default="faiss_index", help="Path to FAISS index directory")
    p.add_argument("--k1", type=int, default=100, help="# of docs from vector search")
    p.add_argument("--k2", type=int, default=10, help="# of docs after rerank")
    args = p.parse_args()

    ans, docs = answer_query(args.query, args.index, k1=args.k1, k2=args.k2)

    sep = "-" * 80
    print(sep)
    print("Answer:\n", textwrap.fill(ans, 100))
    print(sep)
    print("Used documents (metadata + first 120 chars):")
    for d in docs:
        meta = d.metadata.get("timestamp", "?")
        snippet = d.page_content[:120].replace("\n", " ")
        print(f"• ts={meta} | {snippet} …")
