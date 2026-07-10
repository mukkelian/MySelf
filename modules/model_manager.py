"""
model_manager.py
----------------
Loads a Hugging Face model + tokenizer that already lives on the user's
laptop (a folder produced by `save_pretrained`, or a `git clone` of a HF
repo, or a folder downloaded with `huggingface-cli download`).

We keep ONE model + tokenizer cached in memory at a time (`_CACHE`) so the
dashboard does not reload multi-GB weights on every chat message.
"""

import os
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

_CACHE = {"path": None, "device": None, "model": None, "tokenizer": None}

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable, helpful personal assistant. Answer formally and "
    "thoroughly. Use the conversation so far to stay on topic: if the user "
    "asks you to continue, expand on, or finish a previous answer, keep "
    "writing that same thing rather than switching subjects."
)

MAX_HISTORY_TURNS = 6

class _StopOnEvent(StoppingCriteria):
    """Lets /api/chat/stop interrupt a generation already in progress: HF
    checks this once per generated token, so setting the event stops the
    loop after the token currently being produced rather than waiting for
    max_new_tokens."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.stop_event.is_set()


def is_valid_model_folder(path: str) -> bool:
    """A folder is a usable HF model if it has a config.json in it."""
    return os.path.isfile(os.path.join(path, "config.json"))


def get_device(preference: str = "auto") -> str:
    """Resolve a device preference ("auto" / "cpu" / "gpu") to a torch device string.

    "auto" keeps the original behaviour (best available device). "gpu" is
    strict: it raises if no CUDA/MPS device exists, rather than silently
    falling back to CPU, so a user who explicitly asked for GPU finds out.
    """
    if preference == "cpu":
        return "cpu"

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    if preference == "gpu":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        raise ValueError("GPU was requested but no CUDA/MPS device is available on this machine.")

    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"


def load_model(model_path: str, device_preference: str = "auto"):
    """Load (or reuse the cached) tokenizer + model from a local folder."""
    if not is_valid_model_folder(model_path):
        raise ValueError(
            f"'{model_path}' does not look like a Hugging Face model folder "
            "(no config.json found inside it)."
        )

    device = get_device(device_preference)

    if _CACHE["path"] == model_path and _CACHE["device"] == device and _CACHE["model"] is not None:
        return _CACHE["model"], _CACHE["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    ).to(device)

    _CACHE.update({"path": model_path, "device": device, "model": model, "tokenizer": tokenizer})
    return model, tokenizer


def clear_cache():
    """Drop the cached model, e.g. after fine-tuning replaces it on disk."""
    _CACHE.update({"path": None, "device": None, "model": None, "tokenizer": None})


def generate(
    model_path: str,
    question: str,
    context: str | None = None,
    history: list | None = None,
    max_new_tokens: int = 400,
    max_history_turns: int = MAX_HISTORY_TURNS,
    device_preference: str = "auto",
    stop_event: threading.Event | None = None,
) -> str:
    """Generate an answer to `question` (optionally grounded in RAG `context`),
    continuing the conversation in `history` (a list of {"question", "answer"}
    dicts for turns already completed this session -- see app.py's /api/chat).

    `max_history_turns` caps how many of the most recent turns are actually
    sent (0 disables memory entirely); `max_new_tokens` caps the reply
    length. Both are user-configurable in the Chat panel's memory settings.

    Instruct-tuned models (SmolLM2-Instruct, TinyLlama-Chat, ...) only give a
    real answer when the prompt matches the chat format they were trained on;
    feeding them a bare "Question: ...\\nAnswer:" string makes them emit their
    end-of-turn token immediately, i.e. an empty reply. So whenever the
    tokenizer ships a chat template we use it; base/completion models with no
    chat_template fall back to the plain Q/A prompt.
    """
    model, tokenizer = load_model(model_path, device_preference)
    device = get_device(device_preference)

    history = (history or [])[-max_history_turns:] if max_history_turns > 0 else []

    user_content = (
        f"Use the following context to answer.\n\nContext:\n{context}\n\nQuestion: {question}"
        if context else question
    )

    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": user_content})

        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
    else:
        turns = [f"Question: {turn['question']}\nAnswer: {turn['answer']}" for turn in history]
        current = (
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:" if context
            else f"Question: {question}\nAnswer:"
        )
        prompt = "\n\n".join([DEFAULT_SYSTEM_PROMPT] + turns + [current])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)

    stopping_criteria = (
        StoppingCriteriaList([_StopOnEvent(stop_event)]) if stop_event is not None else None
    )

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )
    prompt_len = encoded["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
