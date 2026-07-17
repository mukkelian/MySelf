"""
Saves your chat conversation to disk, locked behind a key you choose - not
even someone else with access to this computer's files can read it without
that key. You're asked for the key every time you start chatting. If you
forget it, there's no way to recover the old conversation - clearing the
chat is the only way forward, and the next key you enter starts a fresh one.
"""

import base64
import json
import os
from threading import RLock

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import config
from . import messages

HISTORY_PATH = os.path.join(config.PROJECT_ROOT, "chat_history.enc")

_lock = RLock()
_fernet = None  # set once a key has been unlocked; cleared again on lock/clear
_salt = None

PBKDF2_ITERATIONS = 390_000


def _derive_key(key: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=PBKDF2_ITERATIONS)
    return base64.urlsafe_b64encode(kdf.derive(key.encode("utf-8")))


def is_unlocked() -> bool:
    return _fernet is not None


def exists() -> bool:
    """Whether there's a saved conversation to unlock, without needing the
    key - lets the dashboard ask for a new key or an existing one by name."""
    return os.path.exists(HISTORY_PATH)


def lock() -> None:
    """Forget the key, so the next chat needs a fresh one."""
    global _fernet, _salt
    with _lock:
        _fernet = None
        _salt = None


def unlock(key: str) -> list:
    """Try a key against the saved conversation and return it decrypted. A
    wrong key raises ValueError. If nothing is saved yet, this key starts a
    brand new, empty conversation instead."""
    global _fernet, _salt
    if not key or not key.strip():
        raise ValueError(messages.CHAT_KEY_REQUIRED)

    with _lock:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                envelope = json.load(f)
            salt = base64.urlsafe_b64decode(envelope["salt"])
            candidate = Fernet(_derive_key(key, salt))
            try:
                history = json.loads(candidate.decrypt(envelope["data"].encode("utf-8")))
            except InvalidToken:
                raise ValueError(messages.CHAT_KEY_INCORRECT)
            _fernet, _salt = candidate, salt
            return history

        salt = os.urandom(16)
        _fernet, _salt = Fernet(_derive_key(key, salt)), salt
        _save([])
        return []


def _save(history: list) -> None:
    token = _fernet.encrypt(json.dumps(history, ensure_ascii=False).encode("utf-8"))
    envelope = {
        "salt": base64.urlsafe_b64encode(_salt).decode("ascii"),
        "data": token.decode("ascii"),
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(envelope, f)


def load() -> list:
    """Return the current conversation. Needs unlock() first."""
    if _fernet is None:
        raise ValueError(messages.CHAT_LOCKED)
    with _lock:
        if not os.path.exists(HISTORY_PATH):
            return []
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            envelope = json.load(f)
        return json.loads(_fernet.decrypt(envelope["data"].encode("utf-8")))


def append(turn: dict) -> None:
    if _fernet is None:
        raise ValueError(messages.CHAT_LOCKED)
    with _lock:
        history = load()
        history.append(turn)
        _save(history)


def clear() -> None:
    """Delete the saved conversation and forget the key - the next chat needs a new one."""
    with _lock:
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
    lock()
