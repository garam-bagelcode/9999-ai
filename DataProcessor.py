# data_processing_chunking.py
"""
RAG preprocessing script

This script loads a Slack export JSON, flattens each thread into a single document,
optionally chunks the documents with token‑level overlap, and writes the result
as a list of {"timestamp", "text"} objects to a JSON file.

Usage (defaults assume the JSON is in the working directory):

    python data_processing_chunking.py --input cvs-economy_final.json --output slack_chunks.json

Requirements:
    pip install langchain tiktoken tqdm python-dotenv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Iterable, List, Tuple

from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_slack_markup(text: str) -> str:
    """Remove basic Slack markup and HTML escapes."""
    text = re.sub(r"[*_~]", "", text)  # bold/italic/strike markup
    return (
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .strip()
    )


def _merge_message(msg: dict) -> str:
    """Return a cleaned message body plus any attached file summaries."""
    parts: List[str] = []

    main_text = msg.get("text", "") or ""
    if main_text:
        parts.append(_clean_slack_markup(main_text))

    # If files are attached, keep their summaries – this is often valuable context
    for f in msg.get("files", []):
        summary = f.get("summary") or ""
        if summary:
            parts.append(f"Attached file summary: {summary}")

    return "\n".join(parts)


def _flatten_messages(messages: List[dict]) -> Iterable[Tuple[str, str]]:
    """Yield (timestamp, concatenated_text) for each root message thread."""
    for msg in messages:
        msg_id = str(msg.get("timestamp") or "")
        if not msg_id:
            continue  # skip if no timestamp

        sections: List[str] = [_merge_message(msg)]

        # Include thread replies (if any) in chronological order
        for reply in msg.get("thread_replies", []):
            sections.append(_merge_message(reply))

        yield msg_id, "\n\n".join(filter(None, sections))


def _chunk_documents(docs: Iterable[Tuple[str, str]], *,
                     chunk_size: int = 800,
                     chunk_overlap: int = 100,
                     encoding_name: str = "cl100k_base") -> List[dict]:
    """Split each document into overlapping token chunks."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked: List[dict] = []
    for ts, text in tqdm(docs, desc="Chunking"):
        for chunk in splitter.split_text(text):
            chunked.append({"timestamp": ts, "text": chunk})
    return chunked

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        messages = json.load(f)

    # Ensure chronological order (optional but nice to have)
    messages.sort(key=lambda m: float(m.get("timestamp", 0)))

    flat_docs = list(_flatten_messages(messages))
    chunks = _chunk_documents(flat_docs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(chunks):,} chunks → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slack JSON → chunked JSON for RAG")
    parser.add_argument("--input", default="cvs-economy_final.json",
                        help="Slack export JSON file (default: cvs-economy_final.json)")
    parser.add_argument("--output", default="slack_chunks.json",
                        help="Output path for chunked JSON (default: slack_chunks.json)")
    args = parser.parse_args()

    main(args.input, args.output)
