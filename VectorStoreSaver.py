# build_faiss_store
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("⚠️  OPENAI_API_KEY not found; set it in .env or environment")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json_records(path: Path):
    """Return list of {timestamp, text} records, validating schema."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not all("timestamp" in rec and "text" in rec for rec in data):
        raise ValueError(f"Invalid schema in {path}; each record needs 'timestamp' and 'text'.")
    return data

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_vector_store(input_paths: List[Path], index_out: Path, texts_per_batch: int = 200):
    """Embed chunks in configurable batches and persist FAISS index."""

    # Gather data
    texts, metadatas = [], []
    for p in input_paths:
        for rec in _read_json_records(p):
            texts.append(rec["text"])
            metadatas.append({"timestamp": rec["timestamp"]})

    total = len(texts)
    print(f"⏳ Embedding {total:,} chunks with 'text-embedding-3-large' …")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072,
        openai_api_key=OPENAI_API_KEY,
    )

    # Incremental building to obey token limits
    vectordb = None
    for i in tqdm(range(0, total, texts_per_batch), desc="Embedding batches"):
        batch_texts = texts[i: i + texts_per_batch]
        batch_meta = metadatas[i: i + texts_per_batch]

        if vectordb is None:
            vectordb = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_meta)
        else:
            vectordb.add_texts(batch_texts, metadatas=batch_meta)

    assert vectordb is not None, "No data embedded!"
    vectordb.save_local(str(index_out))
    print(f"✅ Vector store saved to '{index_out}'")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from JSON chunks")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Path(s) to chunked JSON files")
    parser.add_argument("--index_out", default="faiss_index",
                        help="Directory to persist the FAISS index (default: faiss_index)")
    parser.add_argument("--texts_per_batch", type=int, default=200,
                        help="Max #texts to embed per API call (default: 200)")
    args = parser.parse_args()

    build_vector_store(
        [Path(p) for p in args.inputs],
        Path(args.index_out),
        texts_per_batch=args.texts_per_batch,
    )


if __name__ == "__main__":
    main()