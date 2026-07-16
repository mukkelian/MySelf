"""
Handles voice for the Chat panel: turning your speech into text, translating
between languages, and reading replies out loud. Everything runs on your own
computer, with no cloud services involved.

  1. transcribe_dual()      - turns your recording into text, both in the
                               language you spoke and in English.
  2. translate_from_english() - translates the model's English reply into
                               whichever language you want to hear it in.
  3. synthesize_speech()    - turns text into a spoken audio reply, using a
                               natural-sounding voice where available and a
                               simpler backup voice otherwise.
"""

import io
import os
import subprocess
import tempfile
import threading

import numpy as np
import torch
import wave
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import messages, model_manager

# Every language the app supports, and which voice/translation tools to use for each.
LANGUAGES = {
    "en": {"name": "English", "whisper": "en", "marian": None, "espeak": "en"},
    "hi": {"name": "Hindi", "whisper": "hi", "marian": "Helsinki-NLP/opus-mt-en-hi", "espeak": "hi"},
    "es": {"name": "Spanish", "whisper": "es", "marian": "Helsinki-NLP/opus-mt-en-es", "espeak": "es"},
    "fr": {"name": "French", "whisper": "fr", "marian": "Helsinki-NLP/opus-mt-en-fr", "espeak": "fr"},
    "de": {"name": "German", "whisper": "de", "marian": "Helsinki-NLP/opus-mt-en-de", "espeak": "de"},
    "it": {"name": "Italian", "whisper": "it", "marian": "Helsinki-NLP/opus-mt-en-it", "espeak": "it"},
    "nl": {"name": "Dutch", "whisper": "nl", "marian": "Helsinki-NLP/opus-mt-en-nl", "espeak": "nl"},
    "ar": {"name": "Arabic", "whisper": "ar", "marian": "Helsinki-NLP/opus-mt-en-ar", "espeak": "ar"},
    "ru": {"name": "Russian", "whisper": "ru", "marian": "Helsinki-NLP/opus-mt-en-ru", "espeak": "ru"},
    "zh": {"name": "Chinese (Mandarin)", "whisper": "zh", "marian": "Helsinki-NLP/opus-mt-en-zh", "espeak": "cmn"},
    "ja": {"name": "Japanese", "whisper": "ja", "marian": "Helsinki-NLP/opus-mt-en-jap", "espeak": "ja"},
    "vi": {"name": "Vietnamese", "whisper": "vi", "marian": "Helsinki-NLP/opus-mt-en-vi", "espeak": "vi"},
    "id": {"name": "Indonesian", "whisper": "id", "marian": "Helsinki-NLP/opus-mt-en-id", "espeak": "id"},
    "sw": {"name": "Swahili", "whisper": "sw", "marian": "Helsinki-NLP/opus-mt-en-sw", "espeak": "sw"},
    "tr": {"name": "Turkish", "whisper": "tr", "marian": "Helsinki-NLP/opus-mt-en-tr", "espeak": "tr"},
    "pl": {"name": "Polish", "whisper": "pl", "marian": "Helsinki-NLP/opus-mt-en-pl", "espeak": "pl"},
    # Hinglish: replies stay in English but are read aloud with a Hindi voice,
    # since that's closer to how people actually mix the two languages.
    "hinglish": {"name": "Hinglish (Hindi-English mix)", "whisper": "hi", "marian": None, "espeak": "hi"},
}

# Default speech-to-text accuracy level -- a good balance of speed and accuracy.
STT_MODEL_SIZE = "base"
STT_MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")

# Which voice engine reads replies aloud: "auto" picks the best one
# available, or "bark"/"espeak" to always use one specific engine.
TTS_ENGINES = ("auto", "bark", "espeak")

# The natural-sounding voice model. A smaller version is used here for a
# faster, lighter download.
BARK_MODEL_NAME = "suno/bark-small"

# Languages that have a natural-sounding voice available. Any language not
# listed here uses the simpler backup voice instead.
BARK_VOICE_PRESETS = {
    "en": "v2/en_speaker_6",
    "hi": "v2/hi_speaker_3",
    "hinglish": "v2/hi_speaker_3",
    "es": "v2/es_speaker_3",
    "fr": "v2/fr_speaker_3",
    "de": "v2/de_speaker_3",
    "it": "v2/it_speaker_3",
    "ja": "v2/ja_speaker_3",
    "pl": "v2/pl_speaker_3",
    "ru": "v2/ru_speaker_3",
    "tr": "v2/tr_speaker_3",
    "zh": "v2/zh_speaker_3",
}

_STT_CACHE = {}
_TRANSLATE_CACHE = {}
_BARK_CACHE = {"processor": None, "model": None, "device": None}
_bark_lock = threading.Lock()


def language_options() -> list:
    """List of supported languages, for the dashboard's language dropdowns."""
    return [{"code": code, "name": info["name"]} for code, info in LANGUAGES.items()]


def voice_options() -> dict:
    """List the available speech-to-text accuracy levels and voice engines, for the dashboard's dropdowns."""
    return {"stt_model_sizes": list(STT_MODEL_SIZES), "tts_engines": list(TTS_ENGINES)}


def _require_language(language: str) -> None:
    if language not in LANGUAGES:
        raise ValueError(messages.unsupported_language(language))


def _get_stt_model(model_size: str) -> WhisperModel:
    if model_size not in STT_MODEL_SIZES:
        raise ValueError(messages.unsupported_stt_model_size(model_size))
    if model_size not in _STT_CACHE:
        _STT_CACHE[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _STT_CACHE[model_size]


def transcribe_dual(audio_path: str, language: str, model_size: str = STT_MODEL_SIZE) -> tuple:
    """Turn a recording into text, in both the spoken language and English
    (the English version is what gets sent to the model)."""
    _require_language(language)

    model = _get_stt_model(model_size)
    whisper_lang = LANGUAGES[language]["whisper"]

    segments, _ = model.transcribe(audio_path, language=whisper_lang, task="transcribe")
    native_text = " ".join(segment.text.strip() for segment in segments).strip()

    if language == "en":
        return native_text, native_text

    segments, _ = model.transcribe(audio_path, language=whisper_lang, task="translate")
    english_text = " ".join(segment.text.strip() for segment in segments).strip()
    return native_text, english_text


def _get_translator(marian_model: str):
    if marian_model not in _TRANSLATE_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(marian_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(marian_model)
        model.generation_config.max_length = None  # avoids a repeated harmless warning
        _TRANSLATE_CACHE[marian_model] = (tokenizer, model)
    return _TRANSLATE_CACHE[marian_model]


def translate_from_english(text: str, target_language: str, model_override: str | None = None) -> str:
    """Translate English text into another language. English and Hinglish
    are left unchanged, since no translation is needed for those."""
    _require_language(target_language)

    default_marian = LANGUAGES[target_language]["marian"]
    if default_marian is None or not text.strip():
        return text
    marian_model = model_override or default_marian

    tokenizer, model = _get_translator(marian_model)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=512)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _pcm_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    audio_array = np.clip(audio_array, -1.0, 1.0)
    int16_array = (audio_array * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(int16_array.tobytes())
    return buf.getvalue()


def _get_bark(device: str):
    if _BARK_CACHE["model"] is None or _BARK_CACHE["device"] != device:
        from transformers import AutoProcessor, BarkModel

        _BARK_CACHE["processor"] = AutoProcessor.from_pretrained(BARK_MODEL_NAME)
        _BARK_CACHE["model"] = BarkModel.from_pretrained(BARK_MODEL_NAME).to(device)
        _BARK_CACHE["device"] = device
    return _BARK_CACHE["processor"], _BARK_CACHE["model"]


def _synthesize_bark(text: str, language: str, device_preference: str = "auto") -> bytes:
    device = model_manager.get_device(device_preference)
    with _bark_lock:
        processor, model = _get_bark(device)
        voice_preset = BARK_VOICE_PRESETS[language]
        inputs = processor(text, voice_preset=voice_preset).to(device)
        with torch.no_grad():
            audio_array = model.generate(**inputs)
        sample_rate = model.generation_config.sample_rate
    return _pcm_wav_bytes(audio_array.cpu().numpy().squeeze(), sample_rate)


def _synthesize_espeak(text: str, language: str) -> bytes:
    espeak_voice = LANGUAGES[language]["espeak"]
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        try:
            subprocess.run(
                ["espeak-ng", "-v", espeak_voice, "-w", wav_path],
                input=text.encode("utf-8"),
                check=True,
                timeout=30,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(messages.ESPEAK_NOT_INSTALLED) from exc
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def synthesize_speech(text: str, language: str, device_preference: str = "auto", engine: str = "auto") -> bytes:
    """Turn text into spoken audio. "auto" uses the natural-sounding voice
    when available and falls back to the simpler backup voice otherwise;
    "bark" or "espeak" force one specific voice engine."""
    _require_language(language)
    if engine not in TTS_ENGINES:
        raise ValueError(messages.unsupported_tts_engine(engine))
    if not text.strip():
        text = "..."

    if engine == "espeak":
        return _synthesize_espeak(text, language)

    if engine == "bark":
        if language not in BARK_VOICE_PRESETS:
            raise ValueError(messages.no_bark_voice_for(language))
        return _synthesize_bark(text, language, device_preference)

    if language in BARK_VOICE_PRESETS:
        try:
            return _synthesize_bark(text, language, device_preference)
        except Exception as exc:  # noqa: BLE001
            print(f"[speech] Bark synthesis failed ({exc!r}), falling back to espeak-ng.")

    return _synthesize_espeak(text, language)
