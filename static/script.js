// All of the dashboard's behaviour lives in this one file.

const api = {
  get: (url) => fetch(url).then((r) => r.json()),
  post: (url, body) =>
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }).then((r) => r.json()),
};

function setHint(hint, text, kind) {
  hint.textContent = text;
  hint.className = kind ? `hint ${kind}` : "hint";
}

// Saves a setting and shows how it went in the hint text underneath it.
async function postWithHint(hintId, endpoint, body, { pending = MESSAGES.saving, onSuccess } = {}) {
  const hint = document.getElementById(hintId);
  setHint(hint, pending, null);
  const res = await api.post(endpoint, body);
  setHint(hint, res.ok ? (onSuccess ? onSuccess(res) : "Saved.") : res.error, res.ok ? "success" : "error");
  return res;
}

// ---------------- Resizable panels ----------------
// Dragging a divider resizes the panels next to it, and remembers your choice.
function makeHorizontalResizer({ handleId, cssVar, min, max, storageKey, fromRight }) {
  const handle = document.getElementById(handleId);
  if (!handle) return;

  const stored = Number(localStorage.getItem(storageKey));
  if (stored) {
    document.documentElement.style.setProperty(cssVar, `${Math.min(max, Math.max(min, stored))}px`);
  }

  let dragging = false;

  handle.addEventListener("mousedown", (e) => {
    dragging = true;
    handle.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const containerRect = handle.parentElement.getBoundingClientRect();
    const raw = fromRight ? containerRect.right - e.clientX : e.clientX - containerRect.left;
    const value = Math.min(max, Math.max(min, raw));
    document.documentElement.style.setProperty(cssVar, `${value}px`);
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    const value = parseInt(getComputedStyle(document.documentElement).getPropertyValue(cssVar), 10);
    localStorage.setItem(storageKey, value);
  });
}

makeHorizontalResizer({
  handleId: "sidebarResizer",
  cssVar: "--sidebar-width",
  min: 160,
  max: 420,
  storageKey: "myself.sidebarWidth",
  fromRight: false,
});

makeHorizontalResizer({
  handleId: "chatResizer",
  cssVar: "--chat-sidebar-width",
  min: 220,
  max: 560,
  storageKey: "myself.chatSidebarWidth",
  fromRight: true,
});

makeHorizontalResizer({
  handleId: "datasetResizer",
  cssVar: "--dataset-preview-width",
  min: 220,
  max: 640,
  storageKey: "myself.datasetPreviewWidth",
  fromRight: true,
});

// Fine-Tune panel's 3 columns.
makeHorizontalResizer({
  handleId: "ftResizer1",
  cssVar: "--ft-col1-width",
  min: 240,
  max: 700,
  storageKey: "myself.ftCol1Width",
  fromRight: false,
});

makeHorizontalResizer({
  handleId: "ftResizer2",
  cssVar: "--ft-col2-width",
  min: 240,
  max: 700,
  storageKey: "myself.ftCol2Width",
  fromRight: false,
});

// RAG panel's 3 columns.
makeHorizontalResizer({
  handleId: "ragResizer1",
  cssVar: "--rag-col1-width",
  min: 240,
  max: 700,
  storageKey: "myself.ragCol1Width",
  fromRight: false,
});

makeHorizontalResizer({
  handleId: "ragResizer2",
  cssVar: "--rag-col2-width",
  min: 240,
  max: 700,
  storageKey: "myself.ragCol2Width",
  fromRight: false,
});

// ---------------- Sidebar navigation ----------------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const panel = document.getElementById(`panel-${btn.dataset.panel}`);
    panel.classList.add("active");
  });
});

// ---------------- Sidebar status ----------------
function setActiveBanner(dotId, textId, modelPath) {
  const dot = document.getElementById(dotId);
  const text = document.getElementById(textId);
  if (modelPath) {
    dot.classList.add("ok");
    text.textContent = modelPath;
  } else {
    dot.classList.remove("ok");
    text.textContent = "None selected";
  }
}

async function refreshStatus() {
  const settings = await api.get("/api/settings");

  if (datasetTargetPath === null && settings.dataset_target_file) {
    setDatasetTargetPath(settings.dataset_target_file);
    await refreshDatasetPreview();
  }
  applyDatasetPreviewFontSize(settings.dataset_preview_font_size || 14);

  setActiveBanner("ftActiveDot", "ftActiveModelText", settings.finetune.model_path);
  setActiveBanner("ragActiveDot", "ragActiveModelText", settings.rag.model_path);
  setActiveBanner("chatActiveDot", "chatActiveModelText", settings.chat_model_path);

  if (settings.finetune.model_path) document.getElementById("ftModelPathInput").value = settings.finetune.model_path;
  if (settings.rag.model_path) document.getElementById("ragModelPathInput").value = settings.rag.model_path;
  if (settings.chat_model_path) document.getElementById("chatModelPathInput").value = settings.chat_model_path;

  const ftDataset = settings.finetune.dataset_path;
  const ragDataset = settings.rag.dataset_path;
  if (ftDataset) document.getElementById("ftDatasetPathInput").value = ftDataset;
  if (ragDataset) document.getElementById("ragDatasetPathInput").value = ragDataset;

  document.getElementById("ftEpochs").value = settings.finetune.epochs;
  document.getElementById("ftLr").value = settings.finetune.learning_rate;
  document.getElementById("ftBatch").value = settings.finetune.batch_size;
  document.getElementById("ftMaxLen").value = settings.finetune.max_length;
  document.getElementById("ftLoraR").value = settings.finetune.lora_r;
  document.getElementById("ftLoraAlpha").value = settings.finetune.lora_alpha;
  document.getElementById("ftFullFinetune").checked = !!settings.finetune.full_finetune;
  document.getElementById("ftDevice").value = settings.finetune.device || "auto";

  document.getElementById("ragEmbed").value = settings.rag.embedding_model;
  document.getElementById("ragTopK").value = settings.rag.top_k;
  document.getElementById("ragDevice").value = settings.rag.device || "auto";

  document.getElementById("chatDevice").value = settings.chat_device || "auto";
  document.getElementById("chatHistoryTurns").value = settings.chat_history_turns ?? 6;
  document.getElementById("chatMaxTokens").value = settings.chat_max_new_tokens ?? 400;
  audioModeEnabled = settings.chat_audio_mode ?? true;
  document.getElementById("chatAudioMode").checked = audioModeEnabled;
  document.getElementById("chatSttLanguage").value = settings.chat_stt_language || "en";
  document.getElementById("chatTtsLanguage").value = settings.chat_tts_language || "en";
  document.getElementById("chatSttModel").value = settings.chat_stt_model_size || "base";
  document.getElementById("chatTtsEngine").value = settings.chat_tts_engine || "auto";
  document.getElementById("chatTranslateModel").value = settings.chat_translate_model || "";
  applyChatFontSize(settings.chat_font_size || 15);

  const totalThreads = settings.total_threads || 1;
  const physicalCores = settings.physical_cores || totalThreads;
  populateCpuThreadsSelect(totalThreads, physicalCores);
  document.getElementById("cpuThreads").value = settings.cpu_threads ?? "auto";
  document.getElementById("cpuThreadsHint").textContent =
    `${pluralize(totalThreads, "thread")} available on this machine.`;
}

// ---------------- Native OS file/folder picker ----------------
// "folder" is a plain folder picker (model folders). "dataset" browses
// through files instead (so you can actually see what's in each folder,
// filtered to Q&A/text file types) and uses the folder that held whichever
// file you picked - the same Browse behaviour everywhere in MySelf.
const BROWSE_ENDPOINTS = {
  folder: "/api/fs/pick_folder",
  dataset: "/api/fs/pick_dataset_folder",
};

function wireBrowseButton(browseBtnId, inputId, kind = "folder") {
  document.getElementById(browseBtnId).addEventListener("click", async () => {
    const inputEl = document.getElementById(inputId);
    const res = await api.get(`${BROWSE_ENDPOINTS[kind]}?path=${encodeURIComponent(inputEl.value || "")}`);
    if (res.path) {
      inputEl.value = res.path;
      inputEl.dispatchEvent(new Event("change")); // so fields that save-on-change pick it up right away
    }
  });
}

function formatDatasetSummary(res) {
  const parts = [];
  if (res.count) parts.push(pluralize(res.count, "Q&A pair"));
  if (res.text_chunk_count) parts.push(pluralize(res.text_chunk_count, "text chunk"));
  return parts.length ? `Loaded ${parts.join(" and ")}.` : MESSAGES.noPairsOrTextInFolder;
}

// Shared by the Fine-Tune, RAG, and Chat panels: pick a folder, save it, show a message.
function wirePathSelect({ inputId, browseBtnId, selectBtnId, hintId, endpoint, onSuccess, browseKind }) {
  wireBrowseButton(browseBtnId, inputId, browseKind);

  document.getElementById(selectBtnId).addEventListener("click", async () => {
    const path = document.getElementById(inputId).value.trim();
    await postWithHint(hintId, endpoint, { path }, {
      pending: MESSAGES.checking,
      onSuccess: onSuccess || ((res) => `Model set: ${res.model_path}`),
    });
    refreshStatus();
  });
}

wirePathSelect({
  inputId: "ftModelPathInput",
  browseBtnId: "browseFtModelBtn",
  selectBtnId: "selectFtModelBtn",
  hintId: "ftModelHint",
  endpoint: "/api/finetune/select_model",
});

wirePathSelect({
  inputId: "ftDatasetPathInput",
  browseBtnId: "browseFtDatasetBtn",
  selectBtnId: "selectFtDatasetBtn",
  hintId: "ftDatasetHint",
  endpoint: "/api/finetune/select_dataset",
  onSuccess: formatDatasetSummary,
  browseKind: "dataset",
});

wirePathSelect({
  inputId: "ragModelPathInput",
  browseBtnId: "browseRagModelBtn",
  selectBtnId: "selectRagModelBtn",
  hintId: "ragModelHint",
  endpoint: "/api/rag/select_model",
});

wirePathSelect({
  inputId: "ragDatasetPathInput",
  browseBtnId: "browseRagDatasetBtn",
  selectBtnId: "selectRagDatasetBtn",
  hintId: "ragDatasetHint",
  endpoint: "/api/rag/select_dataset",
  onSuccess: formatDatasetSummary,
  browseKind: "dataset",
});

wirePathSelect({
  inputId: "chatModelPathInput",
  browseBtnId: "browseChatModelBtn",
  selectBtnId: "selectChatModelBtn",
  hintId: "chatModelHint",
  endpoint: "/api/chat/select_model",
  onSuccess: (res) => {
    clearChat(); // switching models starts a fresh conversation
    return res.rag_detected
      ? `Model set: ${res.model_path} — RAG index found, will use retrieval.`
      : `Model set: ${res.model_path} — no RAG index found, will generate directly.`;
  },
});

// Dataset: which Q&A file is currently selected, and which pair (if any) is being edited.
let datasetTargetPath = null;
let datasetEditingIndex = null;

function setDatasetTargetPath(path) {
  datasetTargetPath = path || null;
  document.getElementById("datasetTargetFileHint").textContent =
    datasetTargetPath ? `Target file: ${datasetTargetPath}` : MESSAGES.noFileSelected;
}

function renderDatasetFilePreview(res) {
  const preview = document.getElementById("datasetPreview");
  preview.innerHTML = "";
  if (!datasetTargetPath) {
    preview.textContent = MESSAGES.noFileToPreview;
  } else if (!res.count) {
    preview.textContent = MESSAGES.noPairsInFile;
  } else {
    res.preview.forEach((pair, index) => {
      const item = document.createElement("div");
      item.className = "dataset-preview-item";
      if (index === datasetEditingIndex) item.classList.add("editing");

      const content = document.createElement("div");
      renderFormattedContent(content, `**Q:** ${pair.question}\n\n**A:** ${pair.answer}`);
      item.appendChild(content);

      const actions = document.createElement("div");
      actions.className = "dataset-preview-item-actions";

      const editBtn = document.createElement("button");
      editBtn.className = "btn";
      editBtn.textContent = "Edit";
      editBtn.addEventListener("click", () => enterDatasetEditMode(index, pair));
      actions.appendChild(editBtn);

      const deleteBtn = document.createElement("button");
      deleteBtn.className = "btn danger";
      deleteBtn.textContent = "Delete";
      deleteBtn.addEventListener("click", () => deleteDatasetPair(index));
      actions.appendChild(deleteBtn);

      item.appendChild(actions);
      preview.appendChild(item);
    });
  }
}

function enterDatasetEditMode(index, pair) {
  datasetEditingIndex = index;

  const questionEl = document.getElementById("datasetQuestion");
  const answerEl = document.getElementById("datasetAnswer");
  questionEl.value = pair.question;
  answerEl.value = pair.answer;
  updateDatasetDraftPreview();
  questionEl.focus();

  document.getElementById("datasetAddBtn").textContent = "Save changes";
  document.getElementById("datasetCancelEditBtn").classList.remove("hidden");
  document.getElementById("datasetAddHint").textContent = "";
  refreshDatasetPreview();
}

function exitDatasetEditMode() {
  datasetEditingIndex = null;
  document.getElementById("datasetAddBtn").textContent = "Add Q&A pair";
  document.getElementById("datasetCancelEditBtn").classList.add("hidden");
}

document.getElementById("datasetCancelEditBtn").addEventListener("click", () => {
  exitDatasetEditMode();
  const questionEl = document.getElementById("datasetQuestion");
  const answerEl = document.getElementById("datasetAnswer");
  questionEl.value = "";
  answerEl.value = "";
  updateDatasetDraftPreview();
  document.getElementById("datasetAddHint").textContent = "";
  refreshDatasetPreview();
});

async function deleteDatasetPair(index) {
  if (!confirm(MESSAGES.confirmDeletePair)) return;

  const res = await postWithHint("datasetAddHint", "/api/dataset/delete", { target_path: datasetTargetPath, index }, {
    onSuccess: (res) => `Deleted. File now has ${pluralize(res.count, "pair")}.`,
  });
  if (res.ok) {
    if (datasetEditingIndex !== null) {
      exitDatasetEditMode();
      document.getElementById("datasetQuestion").value = "";
      document.getElementById("datasetAnswer").value = "";
      updateDatasetDraftPreview();
    }
    renderDatasetFilePreview(res);
  }
}

async function refreshDatasetPreview() {
  const res = datasetTargetPath
    ? await api.get(`/api/dataset/preview?path=${encodeURIComponent(datasetTargetPath)}`)
    : { count: 0, preview: [] };
  renderDatasetFilePreview(res);
}

// Shows the Question/Answer you're currently typing, formatted, in the
// Preview pane on the right - the text boxes themselves always stay plain.
function updateDatasetDraftPreview() {
  const question = document.getElementById("datasetQuestion").value;
  const answer = document.getElementById("datasetAnswer").value;
  const draft = document.getElementById("datasetDraftPreview");

  if (!question.trim() && !answer.trim()) {
    draft.innerHTML = "";
    draft.classList.add("hidden");
    return;
  }
  draft.classList.remove("hidden");
  renderFormattedContent(draft, `**Q:** ${question}\n\n**A:** ${answer}`);
}

document.getElementById("datasetQuestion").addEventListener("input", updateDatasetDraftPreview);
document.getElementById("datasetAnswer").addEventListener("input", updateDatasetDraftPreview);

document.getElementById("datasetBrowseBtn").addEventListener("click", async () => {
  const res = await api.get(`/api/fs/pick_save_file?path=${encodeURIComponent(datasetTargetPath || "")}`);
  if (res.path) {
    setDatasetTargetPath(res.path);
    await refreshDatasetPreview();
    refreshStatus();
  }
});

document.getElementById("datasetAddBtn").addEventListener("click", async () => {
  const questionEl = document.getElementById("datasetQuestion");
  const answerEl = document.getElementById("datasetAnswer");
  const hint = document.getElementById("datasetAddHint");

  const question = questionEl.value.trim();
  const answer = answerEl.value.trim();

  if (!question || !answer) {
    setHint(hint, MESSAGES.questionAndAnswerRequired, "error");
    return;
  }
  if (!datasetTargetPath) {
    setHint(hint, MESSAGES.chooseFileFirst, "error");
    return;
  }

  const editing = datasetEditingIndex !== null;
  const endpoint = editing ? "/api/dataset/update" : "/api/dataset/add";
  const body = editing
    ? { question, answer, target_path: datasetTargetPath, index: datasetEditingIndex }
    : { question, answer, target_path: datasetTargetPath };

  const res = await postWithHint("datasetAddHint", endpoint, body, {
    pending: editing ? "Saving changes..." : MESSAGES.saving,
    onSuccess: (res) => editing
      ? `Changes saved. File now has ${pluralize(res.count, "pair")}.`
      : `Saved to ${res.target_path}. File now has ${pluralize(res.count, "pair")}.`,
  });

  if (res.ok) {
    if (editing) exitDatasetEditMode();
    questionEl.value = "";
    answerEl.value = "";
    updateDatasetDraftPreview();
    renderDatasetFilePreview(res);
  }
});

// Dataset: Preview text size.
const DATASET_PREVIEW_FONT_SIZE_MIN = 1;
const DATASET_PREVIEW_FONT_SIZE_MAX = 40;

function applyDatasetPreviewFontSize(size) {
  document.getElementById("datasetPreviewFontSize").value = size;
  document.getElementById("datasetPreview").style.fontSize = `${size}px`;
}

const datasetPreviewFontSizeSelect = document.getElementById("datasetPreviewFontSize");

for (let size = DATASET_PREVIEW_FONT_SIZE_MIN; size <= DATASET_PREVIEW_FONT_SIZE_MAX; size++) {
  const option = document.createElement("option");
  option.value = size;
  option.textContent = size;
  datasetPreviewFontSizeSelect.appendChild(option);
}

datasetPreviewFontSizeSelect.addEventListener("change", async (e) => {
  const size = Number(e.target.value);
  applyDatasetPreviewFontSize(size);
  await api.post("/api/dataset/preview_font_size", { size });
});

// Fine-tune / RAG training
let pollTimer = null;

function startPolling(logElId, onDone) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const status = await api.get("/api/train/status");
    const logEl = document.getElementById(logElId);
    logEl.textContent = status.logs.length ? status.logs.join("\n") : MESSAGES.idle;
    logEl.scrollTop = logEl.scrollHeight;
    if (!status.running) {
      clearInterval(pollTimer);
      pollTimer = null;
      if (status.error) logEl.textContent += `\n\nERROR: ${status.error}`;
      if (onDone) onDone(status);
      refreshStatus();
    }
  }, 1500);
}

document.getElementById("startFinetuneBtn").addEventListener("click", async () => {
  const body = {
    epochs: Number(document.getElementById("ftEpochs").value),
    learning_rate: Number(document.getElementById("ftLr").value),
    batch_size: Number(document.getElementById("ftBatch").value),
    max_length: Number(document.getElementById("ftMaxLen").value),
    lora_r: Number(document.getElementById("ftLoraR").value),
    lora_alpha: Number(document.getElementById("ftLoraAlpha").value),
    full_finetune: document.getElementById("ftFullFinetune").checked,
    device: document.getElementById("ftDevice").value,
  };
  const logEl = document.getElementById("finetuneLog");
  document.getElementById("ftCompleteCard").classList.add("hidden");
  document.getElementById("ftFinalizeHint").textContent = "";

  const res = await api.post("/api/train/finetune", body);
  if (!res.ok) {
    logEl.textContent = "ERROR: " + res.error;
    return;
  }
  logEl.textContent = MESSAGES.starting;
  startPolling("finetuneLog", (status) => {
    if (status.done && !status.error) {
      document.getElementById("ftCompleteCard").classList.remove("hidden");
    }
  });
});

async function finalizeFinetune(body) {
  const res = await postWithHint("ftFinalizeHint", "/api/train/finetune/finalize", body, {
    onSuccess: (res) => `Saved to ${res.saved_to}.`,
  });
  if (res.ok) {
    document.getElementById("ftCompleteCard").classList.add("hidden");
    refreshStatus();
  }
}

document.getElementById("ftSaveHereBtn").addEventListener("click", () => {
  const destination = document.getElementById("ftSaveDestInput").value.trim();
  if (!destination) {
    setHint(document.getElementById("ftFinalizeHint"), MESSAGES.enterDestinationPath, "error");
    return;
  }
  finalizeFinetune({ destination });
});

document.getElementById("ftReplaceBtn").addEventListener("click", () => finalizeFinetune({ replace: true }));

document.getElementById("startRagBtn").addEventListener("click", async () => {
  const body = {
    embedding_model: document.getElementById("ragEmbed").value.trim(),
    top_k: Number(document.getElementById("ragTopK").value),
    device: document.getElementById("ragDevice").value,
  };
  const logEl = document.getElementById("ragLog");
  const res = await api.post("/api/train/rag", body);
  if (!res.ok) {
    logEl.textContent = "ERROR: " + res.error;
    return;
  }
  logEl.textContent = MESSAGES.starting;
  startPolling("ragLog");
});

// Chat: compute device
document.getElementById("chatDevice").addEventListener("change", async () => {
  const device = document.getElementById("chatDevice").value;
  await postWithHint("chatDeviceHint", "/api/chat/device", { device }, {
    onSuccess: (res) => `Chat will use: ${res.resolved}`,
  });
});

// Chat: conversation memory + response length
function saveChatMemorySetting(field, value) {
  return postWithHint("chatMemoryHint", "/api/chat/memory", { [field]: value }, {
    onSuccess: (res) =>
      `Memory: last ${pluralize(res.history_turns, "turn")}, up to ${res.max_new_tokens} tokens per reply.`,
  });
}

document.getElementById("chatHistoryTurns").addEventListener("change", (e) => {
  saveChatMemorySetting("history_turns", Number(e.target.value));
});

document.getElementById("chatMaxTokens").addEventListener("change", (e) => {
  saveChatMemorySetting("max_new_tokens", Number(e.target.value));
});

// Chat: voice languages -- the language you speak and the language replies
// are read back in can be set independently.
const STT_MODEL_SIZE_LABELS = {
  tiny: "Tiny (~75MB, fastest, least accurate)",
  base: "Base (~145MB, fast, balanced -- default)",
  small: "Small (~500MB, slower, more accurate)",
  medium: "Medium (~1.5GB, slow, most accurate)",
  "large-v3": "Large-v3 (~3GB, slowest, best accuracy)",
};

async function loadSpeechLanguages() {
  const [langData, voiceData] = await Promise.all([
    api.get("/api/speech/languages"),
    api.get("/api/speech/voice_options"),
  ]);

  ["chatSttLanguage", "chatTtsLanguage"].forEach((id) => {
    const select = document.getElementById(id);
    select.innerHTML = "";
    langData.languages.forEach(({ code, name }) => {
      const opt = document.createElement("option");
      opt.value = code;
      opt.textContent = name;
      select.appendChild(opt);
    });
  });

  const sttModelSelect = document.getElementById("chatSttModel");
  sttModelSelect.innerHTML = "";
  voiceData.stt_model_sizes.forEach((size) => {
    const opt = document.createElement("option");
    opt.value = size;
    opt.textContent = STT_MODEL_SIZE_LABELS[size] || size;
    sttModelSelect.appendChild(opt);
  });
}

function wireChatLanguageSelect(selectId, endpoint, label) {
  document.getElementById(selectId).addEventListener("change", async () => {
    const select = document.getElementById(selectId);
    await postWithHint("chatLanguageHint", endpoint, { language: select.value }, {
      onSuccess: () => `${label}: ${select.selectedOptions[0].textContent}.`,
    });
  });
}

wireChatLanguageSelect("chatSttLanguage", "/api/chat/stt_language", "Speak in");
wireChatLanguageSelect("chatTtsLanguage", "/api/chat/tts_language", "Hear replies in");

// Chat: speech-to-text accuracy, voice engine, translation model override
document.getElementById("chatSttModel").addEventListener("change", async () => {
  const select = document.getElementById("chatSttModel");
  await postWithHint("chatVoiceHint", "/api/chat/stt_model", { model_size: select.value }, {
    onSuccess: () => `Speech-to-text model: ${select.selectedOptions[0].textContent}.`,
  });
});

document.getElementById("chatTtsEngine").addEventListener("change", async () => {
  const select = document.getElementById("chatTtsEngine");
  await postWithHint("chatVoiceHint", "/api/chat/tts_engine", { engine: select.value }, {
    onSuccess: () => `Text-to-speech engine: ${select.selectedOptions[0].textContent}.`,
  });
});

wireBrowseButton("browseChatTranslateModelBtn", "chatTranslateModel");
wireBrowseButton("browseRagEmbedBtn", "ragEmbed");

document.getElementById("chatTranslateModel").addEventListener("change", async (e) => {
  await postWithHint("chatTranslateModelHint", "/api/chat/translate_model", { model: e.target.value.trim() }, {
    onSuccess: (res) => (res.model ? `Translation model: ${res.model}.` : MESSAGES.usingDefaultTranslateModel),
  });
});

// Chat: locked behind a key until you unlock it, so the saved conversation
// on disk is never readable (or usable) without it.
let chatUnlocked = false;
let chatIsNewSession = false;

async function showChatLockOverlay() {
  chatUnlocked = false;
  document.getElementById("chatLockOverlay").classList.remove("hidden");
  document.getElementById("chatKeyInput").value = "";
  document.getElementById("chatKeyHint").textContent = "";

  // A saved conversation already exists -> ask for its key to unlock it.
  // Nothing saved yet -> this key is starting a brand new one.
  const { exists } = await api.get("/api/chat/session_exists");
  chatIsNewSession = !exists;
  document.getElementById("chatLockHeading").textContent = exists ? "Enter your chat key" : "Start a new chat";
  document.getElementById("chatLockDescription").textContent = exists
    ? "This key locks your saved conversation on disk. Use the same key each time to keep talking in the same conversation, or clear the chat to wipe it and start fresh with a new key."
    : "Choose a key to lock this new conversation. Enter the same key again next time you want to keep talking in it.";
  document.getElementById("chatUnlockBtn").textContent = exists ? "Unlock chat" : "Start chat";
  document.getElementById("chatForgotKeyBtn").classList.toggle("hidden", !exists);
}

function hideChatLockOverlay() {
  chatUnlocked = true;
  document.getElementById("chatLockOverlay").classList.add("hidden");
}

// Shows a saved conversation's turns as chat bubbles, without re-playing audio.
function renderChatHistory(history) {
  history.forEach((turn) => {
    appendBubble(turn.question_display, "user");
    const bubble = appendBubble("", "bot");
    renderBotContent(bubble, turn.answer_display);
    addCopyButton(bubble, turn.answer_display);
    addSpeakButton(bubble, turn.answer_display);
  });
  const chatLog = document.getElementById("chatLog");
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function unlockChat() {
  const key = document.getElementById("chatKeyInput").value;
  if (!key.trim()) {
    setHint(document.getElementById("chatKeyHint"), MESSAGES.chatKeyRequired, "error");
    return;
  }
  const res = await postWithHint("chatKeyHint", "/api/chat/unlock", { key }, {
    pending: chatIsNewSession ? MESSAGES.starting : MESSAGES.unlocking,
    onSuccess: () => (chatIsNewSession ? "Started." : MESSAGES.unlocked),
  });
  if (res.ok) {
    hideChatLockOverlay();
    document.getElementById("chatLog").innerHTML = "";
    renderChatHistory(res.history || []);
  }
}

// Wipes the saved conversation, on the server and on screen, and asks for a
// new key before you can chat again.
async function clearChat() {
  await api.post("/api/chat/clear", {});
  document.getElementById("chatLog").innerHTML = "";
  showChatLockOverlay();
}

async function forgotChatKey() {
  if (!confirm(MESSAGES.confirmClearChat)) return;
  await clearChat();
}

function appendBubble(text, who) {
  const chatLog = document.getElementById("chatLog");
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${who}`;
  const textDiv = document.createElement("div");
  textDiv.className = "bubble-text";
  textDiv.textContent = text;
  bubble.appendChild(textDiv);
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
  return bubble;
}

marked.use({ breaks: true });

// Plain pasted text often has stray leading spaces on some lines (copied
// from a PDF or a job listing, say). Markdown treats 4+ leading spaces as a
// code block, which turns ordinary pasted paragraphs into an unreadable,
// horizontally-scrolling box. This caps accidental indentation everywhere
// except inside real ``` fenced code blocks, so pasted plain text displays
// the way it was pasted, while genuine code snippets stay untouched.
function capAccidentalIndentation(text) {
  let inFence = false;
  return text
    .split("\n")
    .map((line) => {
      if (/^\s{0,3}(```|~~~)/.test(line)) {
        inFence = !inFence;
        return line;
      }
      return inFence ? line : line.replace(/^[ \t]{4,}/, "   ");
    })
    .join("\n");
}

// Turns Markdown/HTML/LaTeX text into safely-formatted content on the page.
// Used for chat replies, the Dataset preview, and its live Question/Answer previews.
function renderFormattedContent(el, text) {
  el.classList.add("rendered-content");
  el.innerHTML = DOMPurify.sanitize(marked.parse(capAccidentalIndentation(text)));
  if (window.renderMathInElement) {
    renderMathInElement(el, { throwOnError: false });
  }
}

function renderBotContent(bubble, text) {
  renderFormattedContent(bubble.querySelector(".bubble-text"), text);
}

function addCopyButton(bubble, text) {
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.textContent = "Copy";
  btn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied!";
    setTimeout(() => {
      btn.textContent = "Copy";
    }, 1500);
  });
  bubble.appendChild(btn);
}

function autoGrowChatInput() {
  const input = document.getElementById("chatInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

// Only one question can be asked at a time.
let chatBusy = false;

function setChatBusy(busy) {
  chatBusy = busy;
  document.getElementById("chatSendBtn").disabled = busy;
  document.getElementById("chatStopBtn").disabled = !busy;
  document.getElementById("chatMicBtn").disabled = busy;
}

// Every reply bubble has its own Play/Pause button. Only one reply plays at
// a time -- starting a new one pauses whatever else was playing.
let currentAudio = null;
let currentSpeechController = null;

// Used by the Stop button: stops any reply currently playing or being generated.
function stopSpeech() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
  }
  if (currentSpeechController) {
    currentSpeechController.abort();
    currentSpeechController = null;
  }
}

function addSpeakButton(bubble, text) {
  const btn = document.createElement("button");
  btn.className = "speak-btn";
  btn.textContent = "▶ Play";
  let audio = null;

  async function play() {
    if (currentAudio && currentAudio !== audio) currentAudio.pause();
    if (currentSpeechController) currentSpeechController.abort();

    if (audio) {
      currentAudio = audio;
      await audio.play().catch(() => {});
      return;
    }

    const language = document.getElementById("chatTtsLanguage").value || "en";
    const controller = new AbortController();
    currentSpeechController = controller;
    btn.disabled = true;
    btn.textContent = "Generating speech..."; // can take a while on slower computers
    try {
      const res = await fetch("/api/speech/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language }),
        signal: controller.signal,
      });
      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("audio")) {
        const err = await res.json();
        throw new Error(err.error || MESSAGES.speechSynthesisFailed);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      audio = new Audio(url);
      audio.addEventListener("play", () => {
        currentAudio = audio;
        btn.textContent = "⏸ Pause";
      });
      audio.addEventListener("pause", () => {
        btn.textContent = "▶ Play";
      });
      audio.addEventListener("ended", () => {
        btn.textContent = "▶ Play";
        audio.currentTime = 0; // so the next Play replays from the start
      });
      btn.disabled = false;
      await audio.play().catch(() => {}); // browser may block autoplay -- clicking Play still works
    } catch (err) {
      if (err.name !== "AbortError") console.error("TTS error:", err); // AbortError just means it was cancelled, not a real failure
      btn.textContent = "▶ Play";
    } finally {
      btn.disabled = false;
      if (currentSpeechController === controller) currentSpeechController = null;
    }
  }

  btn.addEventListener("click", () => {
    if (audio && !audio.paused) {
      audio.pause();
    } else {
      play();
    }
  });

  bubble.appendChild(btn);
  return { play };
}

// Chat: Audio mode -- On auto-plays every reply; Off waits for the Play button.
let audioModeEnabled = true;

document.getElementById("chatAudioMode").addEventListener("change", async (e) => {
  audioModeEnabled = e.target.checked;
  await api.post("/api/chat/audio_mode", { enabled: audioModeEnabled });
});

// Chat: text size.
const CHAT_FONT_SIZE_MIN = 1;
const CHAT_FONT_SIZE_MAX = 40;

function applyChatFontSize(size) {
  document.getElementById("chatFontSize").value = size;
  document.getElementById("chatLog").style.fontSize = `${size}px`;
}

const chatFontSizeSelect = document.getElementById("chatFontSize");

for (let size = CHAT_FONT_SIZE_MIN; size <= CHAT_FONT_SIZE_MAX; size++) {
  const option = document.createElement("option");
  option.value = size;
  option.textContent = size;
  chatFontSizeSelect.appendChild(option);
}

chatFontSizeSelect.addEventListener("change", async (e) => {
  const size = Number(e.target.value);
  applyChatFontSize(size);
  await api.post("/api/chat/font_size", { size });
});

// Sidebar: lets you cap how many CPU threads the app uses.
function populateCpuThreadsSelect(totalThreads, physicalCores) {
  const select = document.getElementById("cpuThreads");
  select.innerHTML = "";
  const autoOption = document.createElement("option");
  autoOption.value = "auto";
  autoOption.textContent = `Auto (Default=${physicalCores})`;
  select.appendChild(autoOption);
  for (let n = 1; n <= totalThreads; n++) {
    const option = document.createElement("option");
    option.value = n;
    option.textContent = n;
    select.appendChild(option);
  }
}

document.getElementById("cpuThreads").addEventListener("change", async (e) => {
  const threads = e.target.value === "auto" ? null : Number(e.target.value);
  await api.post("/api/system/cpu_threads", { threads });
});

// `question`/`questionDisplay` are only passed when sending a voice message;
// a typed message is read straight from the input box instead.
async function sendChat(question, questionDisplay) {
  if (chatBusy || !chatUnlocked) return;
  const typed = question === undefined;
  const input = document.getElementById("chatInput");
  if (typed) {
    question = input.value.trim();
    if (!question) return;
    input.value = "";
    autoGrowChatInput();
  }

  const userBubble = appendBubble(questionDisplay || question, "user");
  const thinkingBubble = appendBubble("Thinking...", "bot");
  const chatLog = document.getElementById("chatLog");

  setChatBusy(true);
  try {
    const res = await api.post("/api/chat", { question, question_display: questionDisplay || null });
    if (res.ok) {
      if (typed) userBubble.querySelector(".bubble-text").textContent = res.question_display;
      renderBotContent(thinkingBubble, res.answer_display);
      addCopyButton(thinkingBubble, res.answer_display);
      const speak = addSpeakButton(thinkingBubble, res.answer_display);
      if (res.stopped) {
        const note = document.createElement("span");
        note.className = "stopped-note";
        note.textContent = MESSAGES.generationStoppedEarly;
        thinkingBubble.appendChild(note);
      } else if (res.answer_display && audioModeEnabled) {
        speak.play();
      }
    } else {
      thinkingBubble.querySelector(".bubble-text").textContent = `Error: ${res.error}`;
    }
  } finally {
    setChatBusy(false);
  }
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function stopChat() {
  const stopBtn = document.getElementById("chatStopBtn");
  stopBtn.disabled = true;
  stopBtn.textContent = "Stopping...";
  stopSpeech();
  await api.post("/api/chat/stop", {});
  stopBtn.textContent = "Stop";
}

// Voice input: record audio, send it off to be transcribed, then send the result as a normal chat message.
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function toggleMic() {
  if (chatBusy || !chatUnlocked) return;
  const micBtn = document.getElementById("chatMicBtn");

  if (isRecording) {
    isRecording = false;
    micBtn.classList.remove("recording");
    mediaRecorder.stop();
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    alert(MESSAGES.micDenied);
    return;
  }

  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) audioChunks.push(e.data);
  };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    handleRecordedAudio();
  };
  mediaRecorder.start();
  isRecording = true;
  micBtn.classList.add("recording");
  micBtn.textContent = "⏹";
  micBtn.title = "Stop recording";
}

async function handleRecordedAudio() {
  const micBtn = document.getElementById("chatMicBtn");
  const language = document.getElementById("chatSttLanguage").value || "en";
  const blob = new Blob(audioChunks, { type: "audio/webm" });

  micBtn.classList.remove("recording");
  micBtn.disabled = true;
  micBtn.textContent = "...";
  micBtn.title = "Transcribing...";
  try {
    const formData = new FormData();
    formData.append("audio", blob, "speech.webm");
    formData.append("language", language);
    const res = await fetch("/api/speech/transcribe", { method: "POST", body: formData }).then((r) => r.json());
    if (res.ok && res.text_en) {
      sendChat(res.text_en, res.text_display);
    } else if (res.ok) {
      alert(MESSAGES.noSpeechCaught);
    } else {
      alert("Transcription failed: " + res.error);
    }
  } finally {
    micBtn.disabled = false;
    micBtn.textContent = "🎤 Speak";
    micBtn.title = "Speak your question";
  }
}

document.getElementById("chatSendBtn").addEventListener("click", () => sendChat());
document.getElementById("chatStopBtn").addEventListener("click", stopChat);
document.getElementById("chatMicBtn").addEventListener("click", toggleMic);
document.getElementById("clearChatBtn").addEventListener("click", () => {
  if (confirm(MESSAGES.confirmClearChat)) clearChat();
});

document.getElementById("chatUnlockBtn").addEventListener("click", unlockChat);
document.getElementById("chatKeyInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    unlockChat();
  }
});
document.getElementById("chatForgotKeyBtn").addEventListener("click", forgotChatKey);

document.getElementById("chatInput").addEventListener("input", autoGrowChatInput);
document.getElementById("chatInput").addEventListener("keydown", (e) => {
  // Enter sends the message; Shift+Enter adds a new line instead.
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});

// Init
loadSpeechLanguages().then(refreshStatus);
showChatLockOverlay();
