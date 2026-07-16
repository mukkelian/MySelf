"""
Saves your chat conversation to a file, so it's still there after you
refresh the page or restart the app. Cleared automatically when you switch
to a different chat model, or when you click "Clear chat".
"""

import json
import os
from threading import RLock

import config

HISTORY_PATH = os.path.join(config.PROJECT_ROOT, "chat_history.json")

_lock = RLock()


def load() -> list:
    with _lock:
        if not os.path.exists(HISTORY_PATH):
            return []
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)


def append(turn: dict) -> None:
    with _lock:
        history = load()
        history.append(turn)
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def clear() -> None:
    with _lock:
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
