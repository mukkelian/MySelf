"""
rag.py
------
Simple Retrieval-Augmented Generation over a mix of Q&A pairs and raw text
chunks (see data_manager.load_text_chunks):

  1. build_index()  - embeds every entry (Q&A pair or text chunk) with a
                       sentence-transformer and saves the embeddings +
                       entries to disk, tagged with their type so retrieval
                       can format each one appropriately later.
  2. retrieve()      - embeds the user's question, finds the most similar
                       stored entries with plain cosine similarity (no extra
                       vector-database dependency needed for a small
                       personal knowledge base). Q&A pairs and text chunks
                       share one index, so retrieval blends both by
                       similarity rather than treating them separately.
  3. answer()        - builds a prompt with the retrieved context and asks
                       the loaded LLM to answer using it.
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from . import model_manager

_EMBEDDER_CACHE = {"name": None, "device": None, "model": None}


def _get_embedder(name: str, device: str | None = None) -> SentenceTransformer:
    if _EMBEDDER_CACHE["name"] != name or _EMBEDDER_CACHE["device"] != device:
        _EMBEDDER_CACHE["model"] = SentenceTransformer(name, device=device)
        _EMBEDDER_CACHE["name"] = name
        _EMBEDDER_CACHE["device"] = device
    return _EMBEDDER_CACHE["model"]


def build_index(
    qa_pairs: list,
    text_chunks: list,
    embedding_model: str,
    index_dir: str,
    device_preference: str = "auto",
    log=print,
) -> str:
    """Embed every Q&A pair and text chunk and save (embeddings.npy +
    pairs.json) to index_dir. Each entry is tagged with a "type" ("qa" or
    "text") so retrieve()/answer() know how to format it later."""
    entries = [{"type": "qa", "question": p["question"], "answer": p["answer"]} for p in qa_pairs]
    entries += [{"type": "text", "text": t} for t in text_chunks]

    if not entries:
        raise ValueError("No Q&A pairs or text found in the selected dataset folder.")

    device = model_manager.get_device(device_preference)
    os.makedirs(index_dir, exist_ok=True)
    log(f"Loading embedding model {embedding_model} on {device} ...")
    embedder = _get_embedder(embedding_model, device)

    log(f"Embedding {len(qa_pairs)} Q&A pairs and {len(text_chunks)} text chunks ...")
    texts = [f"{e['question']} {e['answer']}" if e["type"] == "qa" else e["text"] for e in entries]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(index_dir, "pairs.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"embedding_model": embedding_model, "device": device}, f)

    log(f"RAG index with {len(entries)} entries saved to {index_dir}")
    return index_dir


def _load_index(index_dir: str):
    embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
    with open(os.path.join(index_dir, "pairs.json"), "r", encoding="utf-8") as f:
        entries = json.load(f)
    with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    for entry in entries:
        entry.setdefault("type", "qa")
    return embeddings, entries, meta["embedding_model"], meta.get("device")


def retrieve(question: str, index_dir: str, top_k: int = 3) -> list:
    """Return the top_k most similar entries (Q&A pairs and/or text chunks) to `question`."""
    embeddings, entries, embedding_model, device = _load_index(index_dir)
    embedder = _get_embedder(embedding_model, device)

    query_vec = embedder.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec  # cosine similarity (vectors are normalized)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [entries[i] for i in top_indices]


def answer(
    question: str,
    model_path: str,
    index_dir: str,
    top_k: int = 3,
    device_preference: str = "auto",
    history: list | None = None,
    max_new_tokens: int = 400,
    max_history_turns: int = model_manager.MAX_HISTORY_TURNS,
    stop_event=None,
) -> str:
    """Retrieve context for `question` and ask the loaded LLM to answer using
    it, continuing the conversation in `history` (see model_manager.generate)."""
    context_entries = retrieve(question, index_dir, top_k)
    lines = [
        f"- Q: {e['question']}\n  A: {e['answer']}" if e["type"] == "qa" else f"- {e['text']}"
        for e in context_entries
    ]
    context = "\n".join(lines)
    return model_manager.generate(
        model_path, question, context=context, history=history,
        max_new_tokens=max_new_tokens, max_history_turns=max_history_turns,
        device_preference=device_preference, stop_event=stop_event,
    )
