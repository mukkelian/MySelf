"""
app.py
------
Run it with:  python app.py
Then open:    http://127.0.0.1:8000
"""

import functools
import os
import shutil
import tempfile
import threading
import traceback
from typing import Literal, Optional

import psutil

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from modules import browse, chat_history, data_manager, finetune, messages, model_manager, rag, speech

app = FastAPI(title="MySelf Dashboard")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

RAG_SUBDIR = "rag_index"

job = {"running": False, "mode": None, "logs": [], "error": None, "done": False}
job_lock = threading.Lock()

chat_lock = threading.Lock()
chat_job = {"active": False, "stop_event": None}

# Only one spoken reply is generated at a time, to keep things running smoothly.
speech_lock = threading.Lock()


def api_errors(fn):
    """Wraps a route so any error it raises comes back as {"ok": False,
    "error": ...} instead of a 500 - lets every route just do its work and
    raise on failure, without repeating the same try/except everywhere."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    return wrapper


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
    except Exception as exc:  # noqa: BLE001
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


class ChatBody(BaseModel):
    question: str
    question_display: str | None = None  # the question in your spoken language, if known


class ChatDeviceBody(BaseModel):
    device: Literal["auto", "cpu", "gpu"]


class ChatMemoryBody(BaseModel):
    history_turns: int | None = None
    max_new_tokens: int | None = None


class ChatAudioModeBody(BaseModel):
    enabled: bool


class ChatFontSizeBody(BaseModel):
    size: int


class DatasetPreviewFontSizeBody(BaseModel):
    size: int


class CpuThreadsBody(BaseModel):
    threads: Optional[int] = None  # blank = let the computer decide automatically


class ChatUnlockBody(BaseModel):
    key: str


class ChatLanguageBody(BaseModel):
    language: str


class ChatSttModelBody(BaseModel):
    model_size: str


class ChatTtsEngineBody(BaseModel):
    engine: str


class ChatTranslateModelBody(BaseModel):
    model: str  # empty resets to the default translation model


class SpeakBody(BaseModel):
    text: str
    language: str = "en"


class DatasetQaBody(BaseModel):
    question: str
    answer: str
    target_path: str  # the Q&A file chosen via Browse


class DatasetUpdateBody(BaseModel):
    question: str
    answer: str
    target_path: str
    index: int  # which pair, in order, this is


class DatasetDeleteBody(BaseModel):
    target_path: str
    index: int


@app.get("/api/settings")
def get_settings():
    settings = config.load_settings()
    # Checked fresh every time, so the CPU-threads option always matches this machine.
    settings["total_threads"] = os.cpu_count() or 1
    settings["physical_cores"] = psutil.cpu_count(logical=False) or settings["total_threads"]
    return settings

@app.get("/api/fs/pick_folder")
def fs_pick_folder(path: str = ""):
    """Open a dialog for picking a folder (used for model folders)."""
    return {"path": browse.pick_folder(path)}

@app.get("/api/fs/pick_save_file")
def fs_pick_save_file(path: str = ""):
    """Open a dialog for picking an existing Q&A file, or creating a new one."""
    return {"path": browse.pick_save_file(path)}

@app.get("/api/fs/pick_dataset_folder")
def fs_pick_dataset_folder(path: str = ""):
    """Open a dialog for picking a dataset folder, showing its Q&A/text files
    so you can confirm you're browsing into the right place."""
    return {"path": browse.pick_folder_by_browsing_files(path)}

@app.get("/api/dataset/preview")
def dataset_preview(path: str = ""):
    """Show the Q&A pairs currently in one file."""
    return data_manager.preview_qa_file(path)

@app.post("/api/dataset/add")
@api_errors
def dataset_add(body: DatasetQaBody):
    data_manager.append_qa_pair_to_file(body.target_path, body.question, body.answer)
    config.update_settings({"dataset_target_file": body.target_path})
    return {"ok": True, "target_path": body.target_path, **data_manager.preview_qa_file(body.target_path)}

@app.post("/api/dataset/update")
@api_errors
def dataset_update(body: DatasetUpdateBody):
    """Save an edited Q&A pair."""
    data_manager.update_qa_pair(body.target_path, body.index, body.question, body.answer)
    return {"ok": True, **data_manager.preview_qa_file(body.target_path)}

@app.post("/api/dataset/delete")
@api_errors
def dataset_delete(body: DatasetDeleteBody):
    data_manager.delete_qa_pair(body.target_path, body.index)
    return {"ok": True, **data_manager.preview_qa_file(body.target_path)}

@app.post("/api/finetune/select_model")
def select_finetune_model(body: PathBody):
    if not model_manager.is_valid_model_folder(body.path):
        return {"ok": False, "error": messages.INVALID_MODEL_FOLDER}
    config.update_settings({"finetune": {"model_path": body.path}})
    return {"ok": True, "model_path": body.path}


@app.post("/api/finetune/select_dataset")
@api_errors
def select_finetune_dataset(body: PathBody):
    summary = data_manager.dataset_summary(body.path)
    config.update_settings({"finetune": {"dataset_path": body.path}})
    return {"ok": True, "dataset_path": body.path, **summary}

@app.post("/api/train/finetune")
def train_finetune(body: FinetuneBody):
    settings = config.load_settings()
    if not settings["finetune"]["model_path"]:
        return {"ok": False, "error": messages.SELECT_MODEL_TO_FINETUNE}
    if not settings["finetune"]["dataset_path"]:
        return {"ok": False, "error": messages.SELECT_DATASET_FIRST}
    if job["running"]:
        return {"ok": False, "error": messages.TRAINING_ALREADY_RUNNING}

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
        return {"ok": False, "error": messages.TRAINING_STILL_RUNNING}
    if mode != "finetune" or not done:
        return {"ok": False, "error": messages.NO_COMPLETED_FINETUNE_RUN}
    if error:
        return {"ok": False, "error": messages.LAST_TRAINING_RUN_FAILED}

    settings = config.load_settings()
    staging_dir = os.path.abspath(settings["finetune"]["staging_dir"])
    if not os.path.isdir(staging_dir):
        return {"ok": False, "error": messages.NO_STAGED_MODEL_FOUND}

    if body.replace:
        destination = settings["finetune"]["model_path"]
        if not destination:
            return {"ok": False, "error": messages.NO_BASE_MODEL_TO_REPLACE}
    else:
        destination = body.destination

    if not destination or not destination.strip():
        return {"ok": False, "error": messages.PROVIDE_DESTINATION_PATH}
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
        return {"ok": False, "error": messages.INVALID_MODEL_FOLDER}
    config.update_settings({"rag": {"model_path": body.path}})
    return {"ok": True, "model_path": body.path}


@app.post("/api/rag/select_dataset")
@api_errors
def select_rag_dataset(body: PathBody):
    summary = data_manager.dataset_summary(body.path)
    config.update_settings({"rag": {"dataset_path": body.path}})
    return {"ok": True, "dataset_path": body.path, **summary}

@app.post("/api/train/rag")
def train_rag(body: RagBody):
    settings = config.load_settings()
    if not settings["rag"]["model_path"]:
        return {"ok": False, "error": messages.SELECT_MODEL_FOR_RAG}
    if not settings["rag"]["dataset_path"]:
        return {"ok": False, "error": messages.SELECT_DATASET_FIRST}
    if job["running"]:
        return {"ok": False, "error": messages.TRAINING_ALREADY_RUNNING}

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
        return {"ok": False, "error": messages.INVALID_MODEL_FOLDER}
    config.update_settings({"chat_model_path": body.path})
    model_manager.clear_cache()
    chat_history.clear()  # a new model shouldn't carry over the old conversation
    rag_detected = os.path.isdir(os.path.join(body.path, RAG_SUBDIR))
    return {"ok": True, "model_path": body.path, "rag_detected": rag_detected}


@app.post("/api/chat/device")
@api_errors
def select_chat_device(body: ChatDeviceBody):
    resolved = model_manager.get_device(body.device)
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

@app.get("/api/chat/session_exists")
def chat_session_exists():
    """Whether there's a saved conversation waiting for its key, so the
    dashboard can ask for a new key or an existing one by name."""
    return {"exists": chat_history.exists()}

@app.post("/api/chat/unlock")
@api_errors
def unlock_chat(body: ChatUnlockBody):
    """Check a key against the saved conversation (or start a new one with
    it, if nothing's saved yet) and hand back the conversation unlocked."""
    history = chat_history.unlock(body.key)
    return {"ok": True, "history": history}


@app.post("/api/chat")
def chat(body: ChatBody):
    if not chat_history.is_unlocked():
        return {"ok": False, "error": messages.CHAT_LOCKED}

    with chat_lock:
        if chat_job["active"]:
            return {"ok": False, "error": messages.ANSWER_IN_PROGRESS}
        stop_event = threading.Event()
        chat_job["active"] = True
        chat_job["stop_event"] = stop_event

    try:
        settings = config.load_settings()
        model_path = settings.get("chat_model_path")

        if not model_path:
            return {"ok": False, "error": messages.SELECT_CHAT_MODEL_FIRST}

        rag_index_dir = os.path.join(model_path, RAG_SUBDIR)
        device_preference = settings.get("chat_device", "auto")
        stt_language = settings.get("chat_stt_language", "en")
        tts_language = settings.get("chat_tts_language", "en")
        translate_model_override = settings.get("chat_translate_model") or None
        max_history_turns = settings.get("chat_history_turns", 6)
        max_new_tokens = settings.get("chat_max_new_tokens", 400)

        question_en = body.question
        # Typed questions have no native-language text yet, so translate one.
        question_display = body.question_display or speech.translate_from_english(
            question_en, stt_language, translate_model_override
        )

        persisted = chat_history.load()
        history = [{"question": t["question_en"], "answer": t["answer_en"]} for t in persisted]

        try:
            if os.path.isdir(rag_index_dir):
                answer_en = rag.answer(
                    question_en, model_path, rag_index_dir, settings["rag"]["top_k"],
                    device_preference=device_preference, history=history,
                    max_history_turns=max_history_turns, max_new_tokens=max_new_tokens,
                    stop_event=stop_event,
                )
            else:
                answer_en = model_manager.generate(
                    model_path, question_en, history=history, device_preference=device_preference,
                    max_history_turns=max_history_turns, max_new_tokens=max_new_tokens,
                    stop_event=stop_event,
                )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

        answer_display = speech.translate_from_english(answer_en, tts_language, translate_model_override)

        chat_history.append({
            "question_en": question_en,
            "question_display": question_display,
            "answer_en": answer_en,
            "answer_display": answer_display,
            "stt_language": stt_language,
            "tts_language": tts_language,
        })

        return {
            "ok": True,
            "question_display": question_display,
            "answer_display": answer_display,
            "stopped": stop_event.is_set(),
        }
    finally:
        with chat_lock:
            chat_job["active"] = False
            chat_job["stop_event"] = None


@app.get("/api/chat/history")
@api_errors
def get_chat_history():
    return {"ok": True, "history": chat_history.load()}


@app.post("/api/chat/clear")
def clear_chat_history():
    chat_history.clear()
    return {"ok": True}


@app.post("/api/chat/stop")
def stop_chat():
    with chat_lock:
        if chat_job["active"] and chat_job["stop_event"] is not None:
            chat_job["stop_event"].set()
            return {"ok": True, "stopped": True}
    return {"ok": True, "stopped": False}


@app.get("/api/speech/languages")
def speech_languages():
    return {"languages": speech.language_options()}


@app.get("/api/speech/voice_options")
def speech_voice_options():
    return speech.voice_options()


@app.post("/api/chat/stt_model")
def set_chat_stt_model(body: ChatSttModelBody):
    """How accurate speech-to-text should be when you use the microphone."""
    if body.model_size not in speech.STT_MODEL_SIZES:
        return {"ok": False, "error": messages.UNSUPPORTED_STT_MODEL_SIZE}
    config.update_settings({"chat_stt_model_size": body.model_size})
    return {"ok": True, "model_size": body.model_size}


@app.post("/api/chat/tts_engine")
def set_chat_tts_engine(body: ChatTtsEngineBody):
    """Which voice engine reads replies out loud."""
    if body.engine not in speech.TTS_ENGINES:
        return {"ok": False, "error": messages.UNSUPPORTED_TTS_ENGINE}
    config.update_settings({"chat_tts_engine": body.engine})
    return {"ok": True, "engine": body.engine}


@app.post("/api/chat/translate_model")
def set_chat_translate_model(body: ChatTranslateModelBody):
    """Use a custom translation model instead of the default one; leave empty to use the default."""
    model = body.model.strip()
    config.update_settings({"chat_translate_model": model})
    return {"ok": True, "model": model}


@app.post("/api/chat/audio_mode")
def set_chat_audio_mode(body: ChatAudioModeBody):
    """Turn automatic read-aloud for replies on or off."""
    config.update_settings({"chat_audio_mode": body.enabled})
    return {"ok": True, "enabled": body.enabled}


@app.post("/api/chat/font_size")
def set_chat_font_size(body: ChatFontSizeBody):
    """Remember the chosen chat text size."""
    if not 1 <= body.size <= 40:
        return {"ok": False, "error": messages.font_size_out_of_range()}
    config.update_settings({"chat_font_size": body.size})
    return {"ok": True, "size": body.size}


@app.post("/api/dataset/preview_font_size")
def set_dataset_preview_font_size(body: DatasetPreviewFontSizeBody):
    """Remember the chosen Dataset preview text size."""
    if not 1 <= body.size <= 40:
        return {"ok": False, "error": messages.font_size_out_of_range()}
    config.update_settings({"dataset_preview_font_size": body.size})
    return {"ok": True, "size": body.size}


@app.post("/api/system/cpu_threads")
def set_cpu_threads(body: CpuThreadsBody):
    """Set how many CPU threads the app uses. Switching back to "auto" fully
    takes effect only after restarting the app."""
    total = os.cpu_count() or 1
    if body.threads is not None and not 1 <= body.threads <= total:
        return {"ok": False, "error": messages.threads_out_of_range(total)}
    config.update_settings({"cpu_threads": body.threads})
    model_manager.apply_cpu_thread_setting(body.threads)
    return {"ok": True, "threads": body.threads}


@app.post("/api/chat/stt_language")
def set_chat_stt_language(body: ChatLanguageBody):
    """Which language you speak when using the microphone."""
    if body.language not in speech.LANGUAGES:
        return {"ok": False, "error": messages.UNSUPPORTED_LANGUAGE}
    config.update_settings({"chat_stt_language": body.language})
    return {"ok": True, "language": body.language}


@app.post("/api/chat/tts_language")
def set_chat_tts_language(body: ChatLanguageBody):
    """Which language replies are translated into and read back in --
    can be different from the language you speak in."""
    if body.language not in speech.LANGUAGES:
        return {"ok": False, "error": messages.UNSUPPORTED_LANGUAGE}
    config.update_settings({"chat_tts_language": body.language})
    return {"ok": True, "language": body.language}


@app.post("/api/speech/transcribe")
def speech_transcribe(audio: UploadFile = File(...), language: str = Form("en")):
    """Turn a voice recording into text, both in the language you spoke and in English."""
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio.file.read())
        tmp_path = tmp.name

    model_size = config.load_settings().get("chat_stt_model_size", speech.STT_MODEL_SIZE)
    try:
        text_display, text_en = speech.transcribe_dual(tmp_path, language, model_size)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {"ok": False, "error": str(exc)}
    finally:
        os.remove(tmp_path)

    return {"ok": True, "text_display": text_display, "text_en": text_en}


@app.post("/api/speech/speak")
def speech_speak(body: SpeakBody):
    """Turn text into a spoken audio reply."""
    settings = config.load_settings()
    device_preference = settings.get("chat_device", "auto")
    tts_engine = settings.get("chat_tts_engine", "auto")
    with speech_lock:
        try:
            audio_bytes = speech.synthesize_speech(body.text, body.language, device_preference, tts_engine)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

    return Response(content=audio_bytes, media_type="audio/wav")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
