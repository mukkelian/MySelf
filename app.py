"""
app.py
------
Run it with:  python app.py
Then open:    http://127.0.0.1:8000
"""

import os
import shutil
import threading
import traceback
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from modules import data_manager, finetune, model_manager, rag

app = FastAPI(title="MySelf Dashboard")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

RAG_SUBDIR = "rag_index"

job = {"running": False, "mode": None, "logs": [], "error": None, "done": False}
job_lock = threading.Lock()

chat_lock = threading.Lock()
chat_job = {"active": False, "stop_event": None}


def job_log(line: str):
    with job_lock:
        job["logs"].append(line)
    print(line)


def run_job(mode: str, target_fn, *args):
    with job_lock:
        job.update({"running": True, "mode": mode, "logs": [], "error": None, "done": False})
    try:
        target_fn(*args, log=job_log)
        with job_lock:
            job["done"] = True
    except Exception as exc:  # noqa: BLE001 - surface any training error to the UI
        traceback.print_exc()
        with job_lock:
            job["error"] = str(exc)
    finally:
        with job_lock:
            job["running"] = False


def reset_job():
    with job_lock:
        job.update({"running": False, "mode": None, "logs": [], "error": None, "done": False})


class PathBody(BaseModel):
    path: str


class FinetuneBody(BaseModel):
    epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    max_length: int | None = None
    text_chunk_size: int | None = None
    full_finetune: bool | None = None
    device: Literal["auto", "cpu", "gpu"] | None = None


class FinetuneFinalizeBody(BaseModel):
    destination: str | None = None
    replace: bool = False


class RagBody(BaseModel):
    embedding_model: str | None = None
    chunk_size: int | None = None
    top_k: int | None = None
    device: Literal["auto", "cpu", "gpu"] | None = None


class ChatTurn(BaseModel):
    question: str
    answer: str


class ChatBody(BaseModel):
    question: str
    history: list[ChatTurn] = []


class ChatDeviceBody(BaseModel):
    device: Literal["auto", "cpu", "gpu"]


class ChatMemoryBody(BaseModel):
    history_turns: int | None = None
    max_new_tokens: int | None = None


class DatasetAddBody(BaseModel):
    question: str
    answer: str
    target_file: str
    create_new: bool = False


@app.get("/api/settings")
def get_settings():
    return config.load_settings()

@app.get("/api/fs/pick_folder")
def fs_pick_folder(path: str = ""):
    return {"path": data_manager.pick_folder(path)}

@app.post("/api/dataset/select")
def select_dataset(body: PathBody):
    try:
        summary = data_manager.dataset_summary(body.path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    config.update_settings({"dataset_path": body.path})
    return {"ok": True, "dataset_path": body.path, **summary}


@app.get("/api/dataset/summary")
def dataset_summary():
    settings = config.load_settings()
    if not settings["dataset_path"]:
        return {"count": 0, "preview": []}
    return data_manager.dataset_summary(settings["dataset_path"])

@app.get("/api/dataset/files")
def dataset_files():
    settings = config.load_settings()
    if not settings["dataset_path"]:
        return {"files": [], "active_file": None}
    try:
        files = data_manager.list_dataset_files(settings["dataset_path"])
    except Exception as exc:  # noqa: BLE001
        return {"files": [], "active_file": None, "error": str(exc)}
    return {"files": files, "active_file": settings.get("dataset_active_file")}


@app.post("/api/dataset/add")
def dataset_add(body: DatasetAddBody):
    settings = config.load_settings()
    if not settings["dataset_path"]:
        return {"ok": False, "error": "Select a dataset folder first."}

    try:
        full_path = data_manager.append_qa_pair(
            settings["dataset_path"], body.target_file, body.question, body.answer, body.create_new
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    config.update_settings({"dataset_active_file": os.path.basename(full_path)})
    summary = data_manager.dataset_summary(settings["dataset_path"])
    return {"ok": True, "saved_to": full_path, **summary}

@app.post("/api/finetune/select_model")
def select_finetune_model(body: PathBody):
    if not model_manager.is_valid_model_folder(body.path):
        return {"ok": False, "error": "That folder does not contain a config.json (not a Hugging Face model)."}
    config.update_settings({"finetune": {"model_path": body.path}})
    return {"ok": True, "model_path": body.path}


@app.post("/api/finetune/select_dataset")
def select_finetune_dataset(body: PathBody):
    try:
        summary = data_manager.dataset_summary(body.path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    config.update_settings({"finetune": {"dataset_path": body.path}})
    return {"ok": True, "dataset_path": body.path, **summary}

@app.post("/api/train/finetune")
def train_finetune(body: FinetuneBody):
    settings = config.load_settings()
    if not settings["finetune"]["model_path"]:
        return {"ok": False, "error": "Select a model to fine-tune first."}
    if not settings["finetune"]["dataset_path"]:
        return {"ok": False, "error": "Select a dataset first."}
    if job["running"]:
        return {"ok": False, "error": "A training job is already running."}

    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    settings = config.update_settings({"finetune": patch})

    qa_pairs = data_manager.load_qa_pairs(settings["finetune"]["dataset_path"])
    params = dict(settings["finetune"])
    text_chunks = data_manager.load_text_chunks(settings["finetune"]["dataset_path"], params["text_chunk_size"])
    params["staging_dir"] = os.path.abspath(params["staging_dir"])
    model_path = params["model_path"]

    thread = threading.Thread(
        target=run_job,
        args=("finetune", finetune.train, model_path, qa_pairs, text_chunks, params),
        daemon=True,
    )
    thread.start()
    return {"ok": True, "started": True}


@app.post("/api/train/finetune/finalize")
def finalize_finetune(body: FinetuneFinalizeBody):
    with job_lock:
        running, mode, done, error = job["running"], job["mode"], job["done"], job["error"]

    if running:
        return {"ok": False, "error": "Training is still running."}
    if mode != "finetune" or not done:
        return {"ok": False, "error": "No completed fine-tuning run to save."}
    if error:
        return {"ok": False, "error": "The last training run failed; nothing to save."}

    settings = config.load_settings()
    staging_dir = os.path.abspath(settings["finetune"]["staging_dir"])
    if not os.path.isdir(staging_dir):
        return {"ok": False, "error": "No staged model found."}

    if body.replace:
        destination = settings["finetune"]["model_path"]
        if not destination:
            return {"ok": False, "error": "No base model to replace."}
    else:
        destination = body.destination

    if not destination or not destination.strip():
        return {"ok": False, "error": "Provide a destination path."}
    destination = os.path.abspath(destination.strip())

    if os.path.isdir(destination):
        shutil.rmtree(destination)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    shutil.move(staging_dir, destination)

    model_manager.clear_cache()
    reset_job()
    return {"ok": True, "saved_to": destination}


@app.post("/api/rag/select_model")
def select_rag_model(body: PathBody):
    if not model_manager.is_valid_model_folder(body.path):
        return {"ok": False, "error": "That folder does not contain a config.json (not a Hugging Face model)."}
    config.update_settings({"rag": {"model_path": body.path}})
    return {"ok": True, "model_path": body.path}


@app.post("/api/rag/select_dataset")
def select_rag_dataset(body: PathBody):
    try:
        summary = data_manager.dataset_summary(body.path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    config.update_settings({"rag": {"dataset_path": body.path}})
    return {"ok": True, "dataset_path": body.path, **summary}

@app.post("/api/train/rag")
def train_rag(body: RagBody):
    settings = config.load_settings()
    if not settings["rag"]["model_path"]:
        return {"ok": False, "error": "Select a model for RAG first."}
    if not settings["rag"]["dataset_path"]:
        return {"ok": False, "error": "Select a dataset first."}
    if job["running"]:
        return {"ok": False, "error": "A training job is already running."}

    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    settings = config.update_settings({"rag": patch})

    qa_pairs = data_manager.load_qa_pairs(settings["rag"]["dataset_path"])
    rag_settings = dict(settings["rag"])
    text_chunks = data_manager.load_text_chunks(settings["rag"]["dataset_path"], rag_settings["chunk_size"])
    model_path = os.path.abspath(rag_settings["model_path"])
    index_dir = os.path.join(model_path, RAG_SUBDIR)

    thread = threading.Thread(
        target=run_job,
        args=(
            "rag", rag.build_index, qa_pairs, text_chunks,
            rag_settings["embedding_model"], index_dir, rag_settings["device"],
        ),
        daemon=True,
    )
    thread.start()
    return {"ok": True, "started": True}


@app.get("/api/train/status")
def train_status():
    with job_lock:
        return dict(job)

@app.post("/api/chat/select_model")
def select_chat_model(body: PathBody):
    if not model_manager.is_valid_model_folder(body.path):
        return {"ok": False, "error": "That folder does not contain a config.json (not a Hugging Face model)."}
    config.update_settings({"chat_model_path": body.path})
    model_manager.clear_cache()
    rag_detected = os.path.isdir(os.path.join(body.path, RAG_SUBDIR))
    return {"ok": True, "model_path": body.path, "rag_detected": rag_detected}


@app.post("/api/chat/device")
def select_chat_device(body: ChatDeviceBody):
    try:
        resolved = model_manager.get_device(body.device)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    config.update_settings({"chat_device": body.device})
    model_manager.clear_cache()
    return {"ok": True, "device": body.device, "resolved": resolved}


@app.post("/api/chat/memory")
def set_chat_memory(body: ChatMemoryBody):
    patch = {}
    if body.history_turns is not None:
        patch["chat_history_turns"] = body.history_turns
    if body.max_new_tokens is not None:
        patch["chat_max_new_tokens"] = body.max_new_tokens
    settings = config.update_settings(patch)
    return {"ok": True, "history_turns": settings["chat_history_turns"], "max_new_tokens": settings["chat_max_new_tokens"]}

@app.post("/api/chat")
def chat(body: ChatBody):
    with chat_lock:
        if chat_job["active"]:
            return {"ok": False, "error": "MySelf is still answering the previous question. Wait for it to finish or stop it first."}
        stop_event = threading.Event()
        chat_job["active"] = True
        chat_job["stop_event"] = stop_event

    try:
        settings = config.load_settings()
        model_path = settings.get("chat_model_path")

        if not model_path:
            return {"ok": False, "error": "Select a model to chat with first."}

        rag_index_dir = os.path.join(model_path, RAG_SUBDIR)
        device_preference = settings.get("chat_device", "auto")
        history = [turn.model_dump() for turn in body.history]
        max_history_turns = settings.get("chat_history_turns", 6)
        max_new_tokens = settings.get("chat_max_new_tokens", 400)

        try:
            if os.path.isdir(rag_index_dir):
                reply = rag.answer(
                    body.question, model_path, rag_index_dir, settings["rag"]["top_k"],
                    device_preference=device_preference, history=history,
                    max_history_turns=max_history_turns, max_new_tokens=max_new_tokens,
                    stop_event=stop_event,
                )
            else:
                reply = model_manager.generate(
                    model_path, body.question, history=history, device_preference=device_preference,
                    max_history_turns=max_history_turns, max_new_tokens=max_new_tokens,
                    stop_event=stop_event,
                )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

        return {"ok": True, "answer": reply, "stopped": stop_event.is_set()}
    finally:
        with chat_lock:
            chat_job["active"] = False
            chat_job["stop_event"] = None


@app.post("/api/chat/stop")
def stop_chat():
    with chat_lock:
        if chat_job["active"] and chat_job["stop_event"] is not None:
            chat_job["stop_event"].set()
            return {"ok": True, "stopped": True}
    return {"ok": True, "stopped": False}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
