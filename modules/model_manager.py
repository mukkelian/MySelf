"""
Loads a model from a folder on your computer and keeps it ready in memory,
so the app doesn't have to reload it every time you send a chat message.
"""

import os
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

import config
from . import messages

def apply_cpu_thread_setting(threads=None):
    """Set how many CPU threads the app is allowed to use. Leaving it blank
    lets the computer pick automatically."""
    if threads is None:
        threads = config.load_settings().get("cpu_threads")
    if threads:
        torch.set_num_threads(threads)


apply_cpu_thread_setting()

_CACHE = {"path": None, "device": None, "model": None, "tokenizer": None}

PROMPTS_DIR = os.path.join(config.PROJECT_ROOT, "prompts")

# Used only if the file below doesn't exist yet - it's then written out as a
# starting point, so editing that file (no code changes needed) is enough to
# change how MySelf behaves.
_DEFAULT_WHOAMI = (
    "You are a knowledgeable, helpful personal assistant. Answer formally and "
    "thoroughly. Use the conversation so far to stay on topic: if the user "
    "asks you to continue, expand on, or finish a previous answer, keep "
    "writing that same thing rather than switching subjects."
)
_DEFAULT_USE_CONTEXT = "Use the following context to answer."


def _read_prompt_file(filename: str, default: str) -> str:
    """Read a user-editable prompt from prompts/<filename>, creating it with
    `default` the first time so there's something to edit. Read fresh every
    call, so edits apply immediately without restarting MySelf."""
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    path = os.path.join(PROMPTS_DIR, filename)
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(default)
        return default
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text or default

MAX_HISTORY_TURNS = 6

class _StopOnEvent(StoppingCriteria):
    """Lets the Stop button interrupt a reply that's already being generated."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.stop_event.is_set()


def is_valid_model_folder(path: str) -> bool:
    """Check whether a folder actually contains a usable model."""
    return os.path.isfile(os.path.join(path, "config.json"))


def get_device(preference: str = "auto") -> str:
    """Figure out whether to run on the CPU or GPU. "auto" picks the best
    available option; "gpu" fails loudly if no GPU is found instead of
    silently falling back to CPU."""
    if preference == "cpu":
        return "cpu"

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    if preference == "gpu":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        raise ValueError(messages.gpu_not_available())

    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"


def load_model(model_path: str, device_preference: str = "auto"):
    """Load a model from a folder, or reuse it if it's already loaded."""
    if not is_valid_model_folder(model_path):
        raise ValueError(messages.invalid_model_folder_detail(model_path))

    device = get_device(device_preference)

    if _CACHE["path"] == model_path and _CACHE["device"] == device and _CACHE["model"] is not None:
        return _CACHE["model"], _CACHE["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
    ).to(device)
    model.generation_config.max_length = None

    _CACHE.update({"path": model_path, "device": device, "model": model, "tokenizer": tokenizer})
    return model, tokenizer


def clear_cache():
    """Forget the currently loaded model, e.g. after it's replaced by a fine-tuned version."""
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
    """Generate a reply to `question`, optionally using RAG `context` and the
    earlier conversation in `history`. Uses whichever prompt style the model
    expects, so replies come out sensible instead of blank."""
    model, tokenizer = load_model(model_path, device_preference)
    device = get_device(device_preference)

    history = (history or [])[-max_history_turns:] if max_history_turns > 0 else []
    system_prompt = _read_prompt_file("WhoAmI.txt", _DEFAULT_WHOAMI)

    user_content = (
        f"{_read_prompt_file('UseContext.txt', _DEFAULT_USE_CONTEXT)}\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
        if context else question
    )

    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system_prompt}]
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
        prompt = "\n\n".join([system_prompt] + turns + [current])
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
