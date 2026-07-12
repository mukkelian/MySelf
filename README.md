# MySelf

A small local dashboard for training a personal LLM on your own question/answer
pairs or simple plain text, using either **fine-tuning (LoRA or full)** or 
**RAG (retrieval)** — your choice, picked and configured entirely from the browser. 
There's no single "active model" setting — each panel (Fine-Tune, RAG, Chat) picks 
its own local model folder.

Everything (backend and frontend) runs entirely on your own machine — no
data, prompts, or models ever leave it, and no build step is needed for the
frontend (plain HTML/CSS/JS).

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
- All settings (model paths, dataset paths, hyperparameters, chat memory
  size, device choices, ...) persist to `settings.json` so they survive
  restarts, and every panel picks its own model folder independently — you
  can fine-tune one model, build RAG on another, and chat with a third.

## 1. Install

```bash
cd myself
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Fine-tuning and RAG both run much faster with a GPU, but everything also works
on CPU with a small model (it will just be slower).

## 2. Get a small model onto your laptop

Pick a small causal LM from Hugging Face and save it locally, for example:

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
AutoModelForCausalLM.from_pretrained(name).save_pretrained('./local_models/tinyllama')
AutoTokenizer.from_pretrained(name).save_pretrained('./local_models/tinyllama')
"
```

Any small local causal-LM folder works the same way (GPT-2, Qwen2.5-0.5B, etc.).

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
4. **Chat** — pick a model at the top of the panel the same way, and pick a
   **Compute device** (Auto / CPU only / GPU only) for generation, the same
   as Fine-Tune. If that model folder has a `rag_index/` next to it (built
   in step 3), MySelf automatically retrieves context and answers with it;
   otherwise it just generates directly. Adjust **conversation memory**
   (how many previous turns are sent as context, 0 to disable) and the
   **max reply length** from the panel's memory settings. Only one question
   is answered at a time — Send is disabled while a reply is generating, and
   **Stop** (next to Send) lights up so you can cut generation short and
   edit your question instead of waiting it out. Click "Copy" under any
   answer to copy it to your clipboard.

All settings live in `settings.json` (created on first run), so they survive
restarts and you can inspect/edit them directly if you want.

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
- Answer quality depends heavily on the model you point Chat at. Small
  models (well under 1B parameters) can lose track of facts stated earlier
  in the conversation or give generic/repetitive answers to unrelated
  short inputs — that's a model capacity limitation, not a bug in the
  history/memory handling. A larger local instruct model (1B+ params)
  will follow multi-turn context noticeably better.
- The filesystem "Browse" buttons list real folders on your machine — this app
  is meant to run locally, not be exposed to the internet.
