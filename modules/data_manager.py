"""
Handles reading, saving, editing, and browsing your Q&A files and notes.
Q&A pairs can be stored as .json, .jsonl, or .csv; plain notes can be .txt
or .md files, and get split into smaller chunks for training/search.
"""

import contextlib
import csv
import json
import os
import shutil
import subprocess

SUPPORTED_EXTENSIONS = (".json", ".jsonl", ".csv")
TEXT_EXTENSIONS = (".txt", ".md")


@contextlib.contextmanager
def _tk_root():
    """A hidden helper window needed to open a native file dialog."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        yield root
    finally:
        root.destroy()


def pick_folder(initial_dir: str = "") -> str | None:
    """Open a folder picker window and return the folder the user chose (or None if they cancelled)."""
    initial_dir = os.path.abspath(initial_dir) if initial_dir and os.path.isdir(initial_dir) else os.path.expanduser("~")

    try:
        from tkinter import filedialog
        with _tk_root():
            selected = filedialog.askdirectory(initialdir=initial_dir, title="Select a folder")
        return selected or None
    except ImportError:
        pass
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

    return None


def _resolve_initial_dir(path: str) -> str:
    """Figure out which folder a file dialog should open in, falling back to the home folder."""
    if path and os.path.isdir(path):
        return os.path.abspath(path)
    if path and os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path))
    return os.path.expanduser("~")


def _ensure_supported_extension(path: str | None) -> str | None:
    """If the user typed a filename with no extension at all, default it to .json."""
    if not path or os.path.splitext(path)[1]:
        return path
    return path + ".json"


def pick_save_file(initial_path: str = "") -> str | None:
    """Open the Browse dialog for choosing or creating a Q&A file."""
    initial_dir = _resolve_initial_dir(initial_path)

    try:
        from tkinter import filedialog
        with _tk_root():
            selected = filedialog.asksaveasfilename(
                initialdir=initial_dir, title="Choose or create a Q&A file",
                filetypes=[("JSON", "*.json"), ("JSON Lines", "*.jsonl"), ("CSV", "*.csv"), ("All files", "*.*")],
                defaultextension=".json",
            )
        return _ensure_supported_extension(selected or None)
    except ImportError:
        pass

    if shutil.which("zenity"):
        result = subprocess.run(
            ["zenity", "--file-selection", "--save", "--title=Choose or create a Q&A file",
             f"--filename={initial_dir}/",
             "--file-filter=JSON (*.json) | *.json",
             "--file-filter=JSON Lines (*.jsonl) | *.jsonl",
             "--file-filter=CSV (*.csv) | *.csv",
             "--file-filter=All Q&A files | *.json *.jsonl *.csv",
             "--file-filter=All files (*) | *"],
            capture_output=True, text=True,
        )
        path = result.stdout.strip() or None if result.returncode == 0 else None
        return _ensure_supported_extension(path)

    if shutil.which("kdialog"):
        result = subprocess.run(
            ["kdialog", "--getsavefilename", initial_dir,
             "*.json|JSON\n*.jsonl|JSON Lines\n*.csv|CSV\n*|All files (*)"],
            capture_output=True, text=True,
        )
        path = result.stdout.strip() or None if result.returncode == 0 else None
        return _ensure_supported_extension(path)

    return None


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
    """Split text into smaller overlapping pieces, so no sentence gets cut off
    right at a chunk boundary and lost."""
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
    """Load every .txt/.md file in the folder and split each into smaller pieces."""
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


def append_qa_pair_to_file(full_path: str, question: str, answer: str) -> None:
    """Add one Q&A pair to the chosen file, creating the file first if it doesn't exist yet."""
    if not question.strip() or not answer.strip():
        raise ValueError("Both question and answer are required.")

    full_path = (full_path or "").strip()
    if not full_path:
        raise ValueError("Choose a file to save to first (Save As or Browse).")

    ext = os.path.splitext(full_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"'{os.path.basename(full_path)}' is not a supported dataset file (.json/.jsonl/.csv).")

    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)

    pair = {"question": question.strip(), "answer": answer.strip()}
    if ext == ".jsonl":
        _append_jsonl_file(full_path, pair)
    elif ext == ".json":
        _append_json_file(full_path, pair)
    elif ext == ".csv":
        _append_csv_file(full_path, pair)


def _load_pairs(full_path: str) -> list:
    ext = os.path.splitext(full_path)[1].lower()
    if ext == ".json":
        return _load_json_file(full_path)
    if ext == ".jsonl":
        return _load_jsonl_file(full_path)
    if ext == ".csv":
        return _load_csv_file(full_path)
    return []


def _write_pairs(full_path: str, pairs: list) -> None:
    """Rewrite the whole file with the given list of pairs (used after editing or deleting one)."""
    ext = os.path.splitext(full_path)[1].lower()
    if ext == ".jsonl":
        with open(full_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
    elif ext == ".json":
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2)
    elif ext == ".csv":
        with open(full_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            for pair in pairs:
                writer.writerow(pair)


def _load_pairs_for_edit(full_path: str) -> tuple[str, list]:
    """Check the file exists and is a supported type, then load its pairs."""
    full_path = (full_path or "").strip()
    if not full_path or not os.path.exists(full_path):
        raise ValueError("That file no longer exists.")

    ext = os.path.splitext(full_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"'{os.path.basename(full_path)}' is not a supported dataset file (.json/.jsonl/.csv).")

    return ext, _load_pairs(full_path)


def update_qa_pair(full_path: str, index: int, question: str, answer: str) -> None:
    """Replace one Q&A pair with edited text."""
    if not question.strip() or not answer.strip():
        raise ValueError("Both question and answer are required.")

    _, pairs = _load_pairs_for_edit(full_path)
    if not 0 <= index < len(pairs):
        raise ValueError("That Q&A pair no longer exists (the file may have changed).")

    pairs[index] = {"question": question.strip(), "answer": answer.strip()}
    _write_pairs(full_path, pairs)


def delete_qa_pair(full_path: str, index: int) -> None:
    """Remove one Q&A pair."""
    _, pairs = _load_pairs_for_edit(full_path)
    if not 0 <= index < len(pairs):
        raise ValueError("That Q&A pair no longer exists (the file may have changed).")

    del pairs[index]
    _write_pairs(full_path, pairs)


def preview_qa_file(full_path: str) -> dict:
    """List the Q&A pairs in one file, for the Preview box. A file that
    doesn't exist yet just shows as empty instead of an error."""
    if not full_path or not os.path.exists(full_path):
        return {"count": 0, "preview": []}

    pairs = _load_pairs(full_path)
    return {"count": len(pairs), "preview": pairs[:200]}


def dataset_summary(dataset_path: str) -> dict:
    """Quick summary of a dataset folder: how many Q&A pairs and text
    chunks it holds, plus a few examples of each."""
    pairs = load_qa_pairs(dataset_path)
    chunks = load_text_chunks(dataset_path)
    return {
        "count": len(pairs),
        "preview": pairs[:5],
        "text_chunk_count": len(chunks),
        "text_preview": chunks[:3],
    }
