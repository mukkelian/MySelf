"""
data_manager.py
---------------
Everything related to dataset folders. A dataset folder can hold two kinds
of files, and both can be mixed in the same folder:

  - Q&A pairs, in .json / .jsonl / .csv:
      [ {"question": "...", "answer": "..."}, ... ]        # a JSON list
      {"question": "...", "answer": "..."}\\n{"question": ...}   # JSONL
      question,answer\\n"...","..."                          # CSV with header

  - raw text, in .txt / .md: any plain-language document. There's no
    question/answer structure to parse here -- load_text_chunks() just
    splits it into overlapping word chunks, which Fine-Tune trains on as
    plain next-token continuation and RAG embeds directly for retrieval.

This module also handles browsing the local filesystem so the dashboard can
let a user pick a folder (browsers cannot see local paths on their own, so
the backend does the listing and the frontend just shows it).
"""

import csv
import json
import os
import shutil
import subprocess

SUPPORTED_EXTENSIONS = (".json", ".jsonl", ".csv")
TEXT_EXTENSIONS = (".txt", ".md")


def pick_folder(initial_dir: str = "") -> str | None:
    """Open the OS's native folder picker on the machine running the server
    and return the chosen path, or None if the user cancelled.

    The dashboard is only ever opened on the same machine it runs on, so
    popping a real file manager dialog (rather than an in-page listing) is
    both possible and the more familiar experience.
    """
    initial_dir = os.path.abspath(initial_dir) if initial_dir and os.path.isdir(initial_dir) else os.path.expanduser("~")

    if shutil.which("zenity"):
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory", "--title=Select a folder",
             f"--filename={initial_dir}/"],
            capture_output=True, text=True,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None

    if shutil.which("kdialog"):
        result = subprocess.run(
            ["kdialog", "--getexistingdirectory", initial_dir],
            capture_output=True, text=True,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(initialdir=initial_dir, title="Select a folder")
    root.destroy()
    return selected or None


def _load_json_file(full_path: str) -> list:
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return [{"question": d["question"], "answer": d["answer"]} for d in data
             if "question" in d and "answer" in d]


def _load_jsonl_file(full_path: str) -> list:
    pairs = []
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "question" in d and "answer" in d:
                pairs.append({"question": d["question"], "answer": d["answer"]})
    return pairs


def _load_csv_file(full_path: str) -> list:
    pairs = []
    with open(full_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "question" in row and "answer" in row:
                pairs.append({"question": row["question"], "answer": row["answer"]})
    return pairs


def load_qa_pairs(dataset_path: str) -> list:
    """Load every Q&A pair found in the given folder (all supported files)."""
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset folder not found: {dataset_path}")

    pairs = []
    for name in sorted(os.listdir(dataset_path)):
        full = os.path.join(dataset_path, name)
        ext = os.path.splitext(name)[1].lower()
        if ext == ".json":
            pairs.extend(_load_json_file(full))
        elif ext == ".jsonl":
            pairs.extend(_load_jsonl_file(full))
        elif ext == ".csv":
            pairs.extend(_load_csv_file(full))

    return pairs


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """Split `text` into overlapping word chunks.

    Word-based (not character-based) so chunk_size roughly tracks token
    count without needing a tokenizer here. `overlap` repeats the tail of
    each chunk at the start of the next one, so a fact split across a chunk
    boundary still appears whole in at least one chunk.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_text_chunks(dataset_path: str, chunk_size: int = 200, overlap: int = 40) -> list:
    """Load every .txt/.md file in the folder and split each into overlapping
    word chunks. Unlike Q&A files, there's no structure to parse -- this is
    for raw documents (notes, articles, transcripts, ...)."""
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset folder not found: {dataset_path}")

    chunks = []
    for name in sorted(os.listdir(dataset_path)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in TEXT_EXTENSIONS:
            continue
        full = os.path.join(dataset_path, name)
        with open(full, "r", encoding="utf-8") as f:
            text = f.read()
        chunks.extend(_chunk_text(text, chunk_size, overlap))

    return chunks


def list_dataset_files(dataset_path: str) -> list:
    """List the supported Q&A files inside a dataset folder (for the 'target file' picker)."""
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset folder not found: {dataset_path}")
    return sorted(
        name for name in os.listdir(dataset_path)
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
    )


def _append_json_file(full_path: str, pair: dict) -> None:
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
    else:
        data = []
    data.append(pair)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _append_jsonl_file(full_path: str, pair: dict) -> None:
    with open(full_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair) + "\n")


def _append_csv_file(full_path: str, pair: dict) -> None:
    is_new = not os.path.exists(full_path) or os.path.getsize(full_path) == 0
    with open(full_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        if is_new:
            writer.writeheader()
        writer.writerow(pair)


def append_qa_pair(dataset_path: str, filename: str, question: str, answer: str, create_new: bool) -> str:
    """Append one pasted Q&A pair to a file inside dataset_path.

    If create_new is True, `filename` is created (forced to .jsonl unless the
    caller already gave it a supported extension). Otherwise it must already
    exist, and the pair is appended in whatever format that file already uses.
    """
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset folder not found: {dataset_path}")
    if not question.strip() or not answer.strip():
        raise ValueError("Both question and answer are required.")

    filename = os.path.basename(filename or "").strip()
    if not filename:
        raise ValueError("A target filename is required.")

    ext = os.path.splitext(filename)[1].lower()
    if create_new:
        if ext not in SUPPORTED_EXTENSIONS:
            filename += ".jsonl"
            ext = ".jsonl"
        full_path = os.path.join(dataset_path, filename)
        if os.path.exists(full_path):
            raise ValueError(f"'{filename}' already exists in the dataset folder.")
    else:
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"'{filename}' is not a supported dataset file (.json/.jsonl/.csv).")
        full_path = os.path.join(dataset_path, filename)
        if not os.path.exists(full_path):
            raise ValueError(f"'{filename}' does not exist in the dataset folder.")

    pair = {"question": question.strip(), "answer": answer.strip()}
    if ext == ".jsonl":
        _append_jsonl_file(full_path, pair)
    elif ext == ".json":
        _append_json_file(full_path, pair)
    elif ext == ".csv":
        _append_csv_file(full_path, pair)

    return full_path


def dataset_summary(dataset_path: str) -> dict:
    """Small preview used by the dashboard: how many Q&A pairs and text
    chunks the folder holds, plus a few examples of each. The chunk count
    here is only informational (chunked at the default size) -- Fine-Tune
    and RAG each re-chunk with their own configured chunk size at train time."""
    pairs = load_qa_pairs(dataset_path)
    chunks = load_text_chunks(dataset_path)
    return {
        "count": len(pairs),
        "preview": pairs[:5],
        "text_chunk_count": len(chunks),
        "text_preview": chunks[:3],
    }
