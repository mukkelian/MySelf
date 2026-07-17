"""
Every error and status message the app can show you lives here, in one
place, instead of being scattered and copy-pasted across the other files.
"""

# Models
INVALID_MODEL_FOLDER = "That doesn't look like a valid model folder."
SELECT_MODEL_TO_FINETUNE = "Select a model to fine-tune first."
SELECT_MODEL_FOR_RAG = "Select a model for RAG first."
SELECT_CHAT_MODEL_FIRST = "Select a model to chat with first."


def invalid_model_folder_detail(path: str) -> str:
    return f"'{path}' does not look like a valid model folder (no config.json found inside it)."


def gpu_not_available() -> str:
    return "GPU was requested but no CUDA/MPS device is available on this machine."


# Datasets
NO_QA_OR_TEXT_FOUND = "No Q&A pairs or text found in the selected dataset folder."
SELECT_DATASET_FIRST = "Select a dataset first."
QUESTION_AND_ANSWER_REQUIRED = "Both question and answer are required."
CHOOSE_FILE_FIRST = "Choose a file to save to first (Save As or Browse)."
FILE_NO_LONGER_EXISTS = "That file no longer exists."
PAIR_NO_LONGER_EXISTS = "That Q&A pair no longer exists (the file may have changed)."


def dataset_folder_not_found(path: str) -> str:
    return f"Dataset folder not found: {path}"


def unsupported_dataset_file(filename: str) -> str:
    return f"'{filename}' is not a supported dataset file (.json/.jsonl/.csv)."


# Training jobs
TRAINING_ALREADY_RUNNING = "A training job is already running."
TRAINING_STILL_RUNNING = "Training is still running."
NO_COMPLETED_FINETUNE_RUN = "No completed fine-tuning run to save."
LAST_TRAINING_RUN_FAILED = "The last training run failed; nothing to save."
NO_STAGED_MODEL_FOUND = "No staged model found."
NO_BASE_MODEL_TO_REPLACE = "No base model to replace."
PROVIDE_DESTINATION_PATH = "Provide a destination path."

# Chat
ANSWER_IN_PROGRESS = "MySelf is still answering the previous question. Wait for it to finish or stop it first."
CHAT_KEY_REQUIRED = "Enter a key to start chatting."
CHAT_KEY_INCORRECT = "That key doesn't match this conversation. Try again, or clear the chat to start over with a new key."
CHAT_LOCKED = "Enter your chat key first."

# Speech, language, and translation
UNSUPPORTED_LANGUAGE = "Unsupported language."
UNSUPPORTED_STT_MODEL_SIZE = "Unsupported STT model size."
UNSUPPORTED_TTS_ENGINE = "Unsupported TTS engine."

ESPEAK_NOT_INSTALLED = (
    "espeak-ng is not installed. Install it with your OS package manager "
    "(e.g. 'sudo apt install espeak-ng' on Debian/Ubuntu) to enable spoken replies."
)


def unsupported_language(language: str) -> str:
    return f"Unsupported language '{language}'."


def unsupported_stt_model_size(size: str) -> str:
    return f"Unsupported STT model size '{size}'."


def unsupported_tts_engine(engine: str) -> str:
    return f"Unsupported TTS engine '{engine}'."


def no_bark_voice_for(language: str) -> str:
    return f'Bark has no voice preset for \'{language}\'; use "auto" or "espeak" instead.'


# General limits
def font_size_out_of_range() -> str:
    return "Font size must be between 1 and 40."


def threads_out_of_range(total: int) -> str:
    return f"Threads must be between 1 and {total}."
