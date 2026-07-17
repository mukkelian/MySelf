# MySelf

A small local dashboard for training a personal LLM on your own question/answer
pairs or simple plain text, using either **fine-tuning (LoRA or full)** or 
**RAG (retrieval)** — your choice, picked and configured entirely from the browser. 
There's no single "active model" setting — each panel (Fine-Tune, RAG, Chat) picks 
its own local model folder.

Everything (backend and frontend) runs entirely on your own machine — no
data, prompts, or models ever leave it, and no build step is needed for the
frontend (plain HTML/CSS/JS).

## Downloading a compatible LLM model

MySelf loads models straight from a local folder (not directly from Hugging
Face), so you first need to save one to disk. Any regular "causal LM" text
model works — chat/instruct models like TinyLlama, Qwen2.5, Phi-3, SmolLM2,
etc. Avoid GGUF/ONNX-only repos (search results tagged "GGUF" or with
`*.gguf` files) — those are for other tools like llama.cpp, not this one.

**Files you need in the model folder**

| File | Needed? | What it's for |
| --- | --- | --- |
| `config.json` | **Required** | MySelf checks for this file to decide whether a folder is a valid model at all. |
| `model.safetensors` (or sharded `model-0000X-of-0000Y.safetensors` + `model.safetensors.index.json`) | **Required** | The model's weights. Prefer safetensors over the older `pytorch_model.bin` if the repo offers both. |
| `tokenizer.json`, `tokenizer_config.json` | **Required** | Turns your text into tokens the model understands, and back into text again. |
| `special_tokens_map.json` | Usually required | Defines things like the "end of reply" token. |
| `vocab.json` / `merges.txt` **or** `tokenizer.model` / `spiece.model` | Required if present | Older/alternate tokenizer formats some repos use instead of `tokenizer.json`. |
| `generation_config.json` | Recommended | The model's suggested default generation settings. |
| `chat_template.jinja` (or a `chat_template` field inside `tokenizer_config.json`) | Recommended for chat/instruct models | Formats your conversation the way the model was trained to expect. Without it, MySelf still works but falls back to a plainer prompt format. |
| Anything else (`README.md`, `.gguf`, `*.onnx`, `tf_model.h5`, `flax_model.msgpack`, images, etc.) | Not needed | Other tools/formats MySelf never reads — skip these to save disk space. |

**How to get them onto your machine** — two options:

1. **Easiest — use the included script.** It asks for the model's Hugging
   Face link, checks the repo's real file list, and downloads exactly the
   files from the table above that repo actually has — nothing more. It
   won't pull in duplicate weight copies some repos keep in subfolders
   (`fp16/`, `onnx/`, `gguf/`, ...), so you don't burn bandwidth on formats
   MySelf never reads:

   ```bash
   python download_model.py
   ```

   It'll prompt for the model link (e.g.
   `https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0`, or just
   `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) and a destination folder (defaults
   to `local_models/<model-name>`), then prints the local path to paste into
   MySelf's model folder field.

2. **Manual** — open the model's "Files and versions" tab on huggingface.co
   and download each file from the table above yourself into one folder, or
   use `huggingface-cli download <repo_id> --local-dir <folder>` to grab the
   whole repo at once.

Either way, once you have a folder on disk, point any panel's model field
(Fine-Tune, RAG, or Chat) at it and click "Use this model."

## Features

- **Dataset panel** — point at a local folder of `.json` / `.jsonl` / `.csv`
  Q&A files (or plain `.txt`/`.md` text), preview what's in it, and add new
  Q&A pairs straight from the browser without leaving the dashboard.
- **Fine-Tune panel** — LoRA (default) or full fine-tuning of a local base
  model on your dataset, with configurable epochs / learning rate / batch
  size / LoRA rank, a device picker (Auto / CPU only / GPU only), live
  training logs, and a safe staging → confirm → save/replace flow so nothing
  on disk changes until you say so.
- **RAG panel** — build a retrieval index (sentence-transformer embeddings +
  cosine similarity, no external vector DB) over the same dataset, saved
  alongside the model folder. Chat automatically uses it when present.
- **Chat panel**:
  - Conversation memory is configurable (0–N previous turns sent as
    context; default 6) and a max-new-tokens cap, both adjustable live from
    the panel.
  - Automatically answers with RAG-retrieved context when the selected
    model has a `rag_index/`, otherwise generates directly.
  - Strictly **one answer at a time**: the backend rejects a new request
    while one is already in flight (single-flight, lock-guarded), and the
    Send button is disabled for the duration so you can't queue up
    multiple overlapping questions.
  - A **Stop** button next to Send is enabled only while a reply is being
    generated (greyed out otherwise). Stopping interrupts generation
    token-by-token on the backend (not just a client-side hide), so you get
    back whatever was generated so far and can immediately edit and
    resubmit your question.
  - Per-panel compute device picker (Auto / CPU only / GPU only), same as
    Fine-Tune.
  - Markdown rendering with a "Copy" button under every answer.
  - **Voice input/output**, with independent input/output languages
    ("Speak in" and "Hear replies in") — useful if you mix languages, e.g.
    speak in Hindi but read/hear replies in English, or vice versa. Click 🎤
    and speak your question in your "Speak in" language; the chat box shows
    it in that language, and the LLM sees an English translation
    internally. Every reply is likewise shown translated into your "Hear
    replies in" language *and* read back out loud in it. Typed questions
    get the same treatment — whatever you type is translated for display
    too, not just voice input. Includes a **Hinglish** option for reply
    audio: instead of a full Hindi translation, it reads the reply's
    original English text with a Hindi-accented voice. See "Voice
    pipeline" below for how it works and which languages are supported.
  - **Audio mode** toggle: on (default), every reply is read aloud
    automatically as soon as it's ready; off, replies stay silent until you
    press the ▶/⏸ button on that bubble yourself. Each bubble has a single
    toggle button (not a "play again" button) — pressing it while audio is
    playing pauses in place and resumes from the same spot rather than
    restarting or overlapping a second copy, and only one bubble's audio
    ever plays at a time.
  - Advanced voice controls, also in the same sidebar card: **speech-to-text
    model size** (faster-whisper `tiny` → `large-v3` — bigger is more
    accurate but slower), **text-to-speech engine** (Auto/Bark/espeak-ng —
    see "Voice pipeline"), and an optional **translation model override**
    (type a Hugging Face model id, or browse to a local model folder, to
    replace the default Helsinki-NLP/opus-mt-en-XX translator).
  - **Stop** halts everything, not just the text: interrupting a reply also
    stops (and cancels, if still being synthesized) any audio tied to it.
  - Adjustable **chat text size** (a 1–40px dropdown) applied to the
    whole chat log at once, next to "Clear chat".
  - **Conversation memory is persisted server-side** (`chat_history.enc`),
    not just held in the browser tab — it survives page reloads and server
    restarts, and the visible chat log restores itself from it on load.
    Cleared automatically when you switch chat models, or via "Clear chat".
  - **Locked behind a key you choose.** Every time you load the dashboard,
    Chat starts locked behind an overlay asking for a key (any word or
    phrase) before you can send a message or see any saved conversation.
    The same key unlocks the same conversation next time; a different key
    (or a fresh start with no saved conversation yet) begins a new one.
    Forgot your key? There's no way to recover the old conversation without
    it — click "Forgot your key? Clear the chat" to wipe it and start over.
    See "Chat encryption" below for how this is implemented.
- **Fixed, non-resizable layout**: each panel (Dataset / Fine-Tune / RAG /
  Chat) always fills the full window, and the Chat panel's settings card
  always fills the full height next to the chat box — nothing is dragged
  out of shape or hidden off-screen by accident. Two hidden vertical
  dividers are still draggable: one between the left navigation and the
  panel content, and one between the chat box and its settings card
  (controls only that card's width).
- All settings (model paths, dataset paths, hyperparameters, chat memory
  size, device choices, ...) persist to `settings.json` so they survive
  restarts, and every panel picks its own model folder independently — you
  can fine-tune one model, build RAG on another, and chat with a third.
- **CPU threads control**, in the left navigation sidebar under the
  dataset/chat-model status dots: a dropdown listing every count from 1 up
  to this machine's total logical threads, plus an "Auto (Default=N)"
  option showing the physical-core count PyTorch defaults to (see "CPU
  thread usage" below). It's one process-wide setting shared by Chat,
  Fine-Tune, and RAG alike, and applies immediately — even to a reply
  already streaming — with one exception noted below.

## 1. Install

```bash
cd myself
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Fine-tuning and RAG both run much faster with a GPU, but everything also works
on CPU with a small model (it will just be slower).

The Chat panel's voice output also needs **espeak-ng** installed as a system
package (it's not a Python package, so `pip` can't install it):

```bash
sudo apt install espeak-ng     # Debian/Ubuntu
brew install espeak-ng         # macOS
```

(Windows: grab an installer from the [espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases).)
Voice *input* (Whisper) and translation (MarianMT) need no extra system
packages — both are pulled in by `requirements.txt` and download their
models automatically on first use, the same way the RAG panel's embedding
model already does.

## 2. Get a small model onto your laptop

Pick a small causal LM from Hugging Face and save it locally. The easiest
way is the included script — it asks for the model's Hugging Face link and
downloads just the files MySelf needs:

```bash
python download_model.py
```

See "Downloading a compatible LLM model" below for exactly which files that
is and a manual alternative. Any small local causal-LM folder works the same
way (TinyLlama, GPT-2, Qwen2.5-0.5B, etc.).

## 3. Prepare your Q&A data

Put one or more `.json`, `.jsonl`, or `.csv` files in a folder. Each entry needs
a `question` and an `answer` field. See `data/example_qa.json` for the format.

Or, train your personal LLM model with the data present in simple plain text file

## 4. Run the dashboard

```bash
python app.py
```

Open **http://127.0.0.1:8000** in your browser.

## 5. Use the dashboard

1. **Dataset** — browse to your Q&A folder, click "Use this dataset." A preview
   appears. You can also build the dataset straight from the browser: paste a
   question and answer into the "Add a Q&A pair" card and either append it to
   an existing file or create a new one — subsequent pairs default to
   whichever file you last used.
2. **Fine-Tune** — browse to (or type) the base model folder to fine-tune,
   click "Use this model." Set epochs / learning rate / LoRA settings, or
   check "Full fine-tuning" to train every parameter instead of LoRA
   adapters. Pick a **Compute device** — "Auto" uses a GPU if one is
   available and falls back to CPU, or you can force "CPU only" / "GPU only"
   (the latter fails fast with a clear error if no GPU is found, instead of
   silently running on CPU). Click "Start fine-tuning." When training
   finishes, a "Training complete" card asks where to save the result —
   enter a new folder path, or click "Replace original model" to overwrite
   the base model folder you fine-tuned in place.
3. **RAG** — browse to (or type) the model folder to use for retrieval,
   click "Use this model." Set the embedding model and top-K, click
   "Build RAG index." The index files are saved into `<model folder>/rag_index/`,
   right alongside that model (no weights change).
4. **Chat** — first, enter a key (any word or phrase) in the overlay to
   unlock the chat box — this both decrypts your saved conversation (if any)
   and locks new messages behind that same key going forward. Everything for
   this panel (model, device, memory, audio mode, voice languages,
   STT/TTS/translation choices) lives in one sidebar card docked to the
   right of the chat box, at a fixed full height (drag the hidden vertical
   line between them to resize its width). Pick a model the same way as the
   other panels; if that model folder has a `rag_index/` next to it (built
   in step 3), MySelf automatically retrieves context and answers with it,
   otherwise it just generates directly. Adjust
   **conversation memory** (how many previous turns are sent as context, 0
   to disable) and the **max reply length**. Only one question is answered
   at a time — Send is disabled while a reply is generating, and **Stop**
   (next to Send) lights up so you can cut generation (and any in-progress
   or playing audio) short and edit your question instead of waiting it
   out. Click "Copy" under any answer to copy it to your clipboard, or the
   ▶/⏸ button to play, pause, or resume its audio — or leave **Audio mode**
   on to have replies read aloud automatically. Pick **Speak in** / **Hear
   replies in** to chat by voice, and optionally tune the **speech-to-text
   model size**, **text-to-speech engine**, and **translation model
   override** underneath — see "Voice pipeline" below. Use the **Text
   size** field above the chat log to resize everything in it.

All settings live in `settings.json` (created on first run), so they survive
restarts and you can inspect/edit them directly if you want.

## Voice pipeline

The Chat panel's voice feature (`modules/speech.py`) chains together
separate, open-source, commercial-use-friendly models/tools — no cloud
speech API is involved, everything runs locally:

1. **Speech → text**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (MIT license; OpenAI's Whisper weights are MIT too), run **twice** per
   recording — once with its "transcribe" task (native-language text, for
   the chat bubble) and once with "translate" (English text, for the LLM).
   Whisper only does one task per pass, so this is how a faithful native
   transcript and an English translation both come out of the same
   recording, without a second translation model. (For English audio both
   passes would be identical, so only one actually runs.) The **model
   size** is user-selectable in the Chat panel (`tiny` → `large-v3`) — the
   default `base` is fast but can mis-transcribe code-switched speech
   (e.g. Hinglish) into the wrong script; stepping up to `small` or bigger
   noticeably improves that at the cost of speed.
2. **Text ↔ your language**: [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP)
   (MarianMT, Apache-2.0) translates English into your language and back —
   used for a typed question's display translation, and for translating
   the LLM's English reply for both display and speech. The Chat panel's
   "Translation model (advanced)" field can override this per-language
   default with any other Hugging Face seq2seq model id or a local model
   folder (browse to one, same as picking a chat model).
3. **Text → speech**: [Bark](https://github.com/suno-ai/bark) (Suno AI,
   MIT license) generates natural speech for the languages it has voice
   presets for. Anything outside that set — or any Bark failure — falls
   back automatically to [espeak-ng](https://github.com/espeak-ng/espeak-ng)
   (GPL-3.0, run as a separate subprocess so it doesn't impose the GPL on
   this codebase, the same pattern most software uses to shell out to
   `ffmpeg`), which covers every language MySelf supports, instantly, if
   more robotic-sounding. The Chat panel's **text-to-speech engine** picker
   lets you force one or the other instead of this automatic fallback.

The LLM itself always thinks and answers in English internally; the chat
box and the audio at the edges are translated. "Speak in" (voice input)
and "Hear replies in" (voice output) are independent settings, so you're
not locked into one language for both directions — e.g. speak in Hindi,
read/hear replies in Hinglish. Supported languages: English, Hindi,
Spanish, French, German, Italian, Dutch, Arabic, Russian, Chinese
(Mandarin), Japanese, Vietnamese, Indonesian, Swahili, Turkish, Polish, and
Hinglish (see `LANGUAGES` in `modules/speech.py` to add more — any language
with an `opus-mt-en-XX` model on the HF Hub and an espeak-ng voice works;
add a `BARK_VOICE_PRESETS` entry too if Bark has a voice for it).

**Hinglish** is a special case: no open model transcribes or translates
code-mixed Hindi-English as its own language, so on the input side it
behaves exactly like Hindi (Whisper's translate task collapses either hint
to English output anyway). The real difference is on reply *audio*: picking
Hinglish as your "Hear replies in" language skips the Hindi translation
step and just has the Hindi voice (Bark preset or espeak-ng fallback) read
the reply's original English text aloud — an approximation of how Hinglish
is actually spoken, rather than a full Hindi translation.

**Performance note on Bark**: it's a full generative model, not a
lookup-table synthesizer like espeak-ng, so on CPU-only hardware expect
anywhere from a few seconds to over a minute per reply (the first call
also downloads ~2GB of weights). It uses the same compute device as the
chat LLM (GPU if configured), which helps a lot if you have one. Whisper
and MarianMT stay CPU-only regardless — both are small/fast enough there,
and it keeps them from competing with the main LLM for GPU memory.

**CPU thread usage**: left alone (the sidebar's CPU threads dropdown on
"Auto"), PyTorch defaults to one thread per *physical* CPU core, not every
logical thread a hyperthreaded/SMT chip reports — e.g. on an 8-core/16-thread
CPU, expect ~8 threads in use during generation, not 16. That's a deliberate
PyTorch default, not a MySelf limitation: dense matrix-multiply workloads
like LLM inference mostly don't benefit from hyperthreads, since sibling
threads share the same core's execution units. The physical-core count
shown in "Auto (Default=N)" is detected per-machine via `psutil`, so it's
accurate on whatever laptop MySelf runs on.

This is fully user-configurable from the dashboard now (see "CPU threads
control" above) rather than requiring an `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
environment variable — pick any thread count up to the machine's logical
total, and it applies process-wide to Chat, Fine-Tune, and RAG immediately,
including to a Chat reply already in progress. The one case that still
needs a restart: switching back to "Auto" after a custom number, since
PyTorch has no API to un-cap its own thread pool once
`torch.set_num_threads()` has been called — that only takes full effect
again on the next `python app.py` start.

**Conversation memory**: every turn (English + translated-display text,
both question and answer) is appended to `chat_history.enc` at the project
root after each reply — see `modules/chat_history.py`. It's encrypted (see
"Chat encryption" below), so the file on disk is unreadable without the key
you unlocked it with, even to someone else who can browse this computer's
files.

**Chat encryption**: the Chat panel starts locked every time you load the
dashboard — you can't send a message or see any saved conversation until
you enter a key (any word or phrase). That key, combined with a random
salt stored alongside the data, derives an AES encryption key via PBKDF2
(390,000 iterations, `cryptography`'s `Fernet`); the whole conversation is
encrypted with it before being written to `chat_history.enc`. The key
itself is never written to disk — only kept in the server's memory for as
long as the chat stays unlocked, and forgotten again on "Clear chat" or a
server restart, which is why you're asked for it every time you start
chatting, not just the first time. There's no password reset: a wrong key
is simply rejected, and the only way past a forgotten one is "Forgot your
key? Clear the chat," which deletes the encrypted file entirely so a new
key can start a fresh conversation. This only covers the chat
conversation — other saved state (`settings.json`, your dataset files,
model folders) is unaffected and stored as before.

## Notes & limits

- Fine-tuning supports both `peft` LoRA adapters (default, fast) and full
  fine-tuning (every parameter, slower, no LoRA settings used). RAG uses a
  plain cosine-similarity search (no external vector DB) — enough for a
  personal, small-scale project without extra infra.
- Fine-tuning always trains into a staging folder first
  (`models/_staging_finetune`) and only moves it to its final destination
  once you confirm a save path or choose "Replace original model" — nothing
  on disk changes until you do.
- Only one training job runs at a time; the dashboard tells you if one is already running.
- Chat is single-flight too: only one question is answered at a time
  (backend-enforced, not just a UI restriction), and generation can be
  interrupted mid-way with the Stop button.
- Speech synthesis is also single-flight on the backend (Bark is heavy
  enough that two at once would just make both slower), and the frontend
  only ever plays one clip at a time — starting a different bubble's audio
  pauses whatever was already playing first. Each bubble's ▶/⏸ button
  toggles and resumes in place rather than restarting from scratch.
- Answer quality depends heavily on the model you point Chat at. Small
  models (well under 1B parameters) can lose track of facts stated earlier
  in the conversation or give generic/repetitive answers to unrelated
  short inputs — that's a model capacity limitation, not a bug in the
  history/memory handling. A larger local instruct model (1B+ params)
  will follow multi-turn context noticeably better.
- The filesystem "Browse" buttons list real folders on your machine — this app
  is meant to run locally, not be exposed to the internet.
