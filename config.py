"""
config.py
---------
Very small helper that reads/writes a single JSON file (settings.json).
Every setting picked on the dashboard (model path, dataset path, training
hyper-parameters, chosen mode, ...) lives in that one file so the whole
app has one source of truth and restarting the server does not lose state.
"""

import json
import os
from threading import RLock

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")

_lock = RLock()

DEFAULTS = {
    "dataset_path": None,          # local folder that contains Q&A files
    "dataset_active_file": None,   # filename (inside dataset_path) that pasted Q&A pairs append to
    "chat_model_path": None,       # model folder currently selected in the Chat panel
    "chat_device": "auto",         # "auto" (GPU if available), "cpu", or "gpu" -- used for chat generation
    "chat_history_turns": 6,       # how many previous Q/A turns to send as context (0 = no memory)
    "chat_max_new_tokens": 400,    # max tokens generated per Chat reply
    "finetune": {
        "model_path": None,        # base model folder selected in the Fine-Tune panel
        "dataset_path": None,      # Q&A folder selected in the Fine-Tune panel
        "full_finetune": False,    # False = LoRA adapters, True = train all parameters
        "device": "auto",          # "auto" (GPU if available), "cpu", or "gpu"
        "epochs": 3,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_length": 256,
        "staging_dir": "models/_staging_finetune",
        "text_chunk_size": 200,  # words per training example when the dataset folder has raw .txt/.md files
    },
    "rag": {
        "model_path": None,        # model folder selected in the RAG panel
        "dataset_path": None,      # Q&A folder selected in the RAG panel
        "device": "auto",          # "auto" (GPU if available), "cpu", or "gpu" -- used to build the index
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 200,
        "top_k": 3,
    },
}


def _with_defaults(settings: dict) -> dict:
    """Fill in any keys missing from an older settings.json with their
    default value, one level deep -- so adding a new setting later never
    breaks an install that already has a settings.json on disk."""
    merged = json.loads(json.dumps(DEFAULTS))
    for key, value in settings.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def load_settings() -> dict:
    """Return current settings, creating the file with defaults if missing."""
    with _lock:
        if not os.path.exists(SETTINGS_PATH):
            save_settings(DEFAULTS)
            return json.loads(json.dumps(DEFAULTS))
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        merged = _with_defaults(settings)
        if merged != settings:
            save_settings(merged)
        return merged


def save_settings(settings: dict) -> None:
    """Persist the given settings dict to disk."""
    with _lock:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)


def update_settings(patch: dict) -> dict:
    """Shallow-merge `patch` into the current settings and save.

    Nested dicts (like 'finetune' or 'rag') are merged one level deep so a
    partial update (e.g. only {'finetune': {'epochs': 5}}) does not wipe
    out the other keys in that section.
    """
    current = load_settings()
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key].update(value)
        else:
            current[key] = value
    save_settings(current)
    return current
