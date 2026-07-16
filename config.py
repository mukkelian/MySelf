"""
Keeps track of all your settings (which model you picked, dataset path,
training options, etc.) in one file called settings.json, so nothing is
lost when you restart the app.
"""

import json
import os
from threading import RLock

PROJECT_ROOT = os.path.dirname(__file__)
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.json")

_lock = RLock()

DEFAULTS = {
    "dataset_target_file": None,   # the Q&A file currently chosen in the Dataset tab
    "dataset_preview_font_size": 14,  # text size in the Dataset preview box
    "chat_model_path": None,       # the model currently chosen for chatting
    "chat_device": "auto",         # "auto", "cpu", or "gpu" -- what runs the chat
    "chat_history_turns": 6,       # how many earlier messages the chat remembers
    "chat_max_new_tokens": 400,    # how long a chat reply can be
    "chat_audio_mode": True,       # True = read replies out loud automatically
    "chat_font_size": 15,          # text size in the chat window
    "chat_stt_language": "en",     # language you speak when using the microphone
    "chat_tts_language": "en",     # language replies are read back in
    "chat_stt_model_size": "base", # how accurate speech-to-text is (bigger = slower, more accurate)
    "chat_tts_engine": "auto",     # which voice engine reads replies aloud
    "chat_translate_model": "",    # custom translation model, blank = use the default
    "cpu_threads": None,           # how many CPU threads to use; blank = let the computer decide
    "finetune": {
        "model_path": None,        # base model chosen in the Fine-Tune tab
        "dataset_path": None,      # Q&A folder chosen in the Fine-Tune tab
        "full_finetune": False,    # True = retrain everything (slower); False = lighter LoRA training
        "device": "auto",          # "auto", "cpu", or "gpu"
        "epochs": 3,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_length": 256,
        "staging_dir": "models/_staging_finetune",
        "text_chunk_size": 200,  # how many words go into one training example
    },
    "rag": {
        "model_path": None,        # model chosen in the RAG tab
        "dataset_path": None,      # Q&A folder chosen in the RAG tab
        "device": "auto",          # "auto", "cpu", or "gpu"
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 200,
        "top_k": 3,
    },
}


def _with_defaults(settings: dict) -> dict:
    """Add any missing settings back in with their default value, so an
    older settings.json still works after the app gains new options."""
    merged = json.loads(json.dumps(DEFAULTS))
    for key, value in settings.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def load_settings() -> dict:
    """Read your saved settings, creating the file with defaults if it doesn't exist yet."""
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
    """Write the given settings to disk."""
    with _lock:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)


def update_settings(patch: dict) -> dict:
    """Update just the given settings and save, without wiping out the rest.

    Grouped settings (like 'finetune' or 'rag') keep their other values --
    e.g. updating only {'finetune': {'epochs': 5}} won't erase the rest of
    that group.
    """
    current = load_settings()
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key].update(value)
        else:
            current[key] = value
    save_settings(current)
    return current
