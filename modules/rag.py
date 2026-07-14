"""
Lets the model look up relevant Q&A pairs and notes before answering, so it
can give better answers without needing to be retrained.

  1. build_index() - reads all your Q&A pairs and notes, and saves them in a
                      searchable form.
  2. retrieve()     - finds the entries most relevant to a question.
  3. answer()       - looks up relevant entries, then asks the model to
                      answer using them.
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
    """Turn every Q&A pair and note into a searchable form and save it to index_dir."""
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
    """Find the entries that best match a question."""
    embeddings, entries, embedding_model, device = _load_index(index_dir)
    embedder = _get_embedder(embedding_model, device)

    query_vec = embedder.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec  # how closely each entry matches the question
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
    """Look up relevant context for `question` and ask the model to answer using it."""
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
