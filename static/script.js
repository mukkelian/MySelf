// ---------------------------------------------------------------------------
// script.js - all dashboard behaviour lives here (no build step, no framework)
// ---------------------------------------------------------------------------

const api = {
  get: (url) => fetch(url).then((r) => r.json()),
  post: (url, body) =>
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }).then((r) => r.json()),
};

// ---------------- Resizable panels ----------------
// Drags a handle to live-update a CSS variable (which the layout's widths
// are defined in terms of -- see --sidebar-width / --chat-sidebar-width in
// style.css), then remembers the chosen size in localStorage.
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

// Resizable boxes (shared by panels AND the cards inside them)
// A resizable box is any absolutely-positioned element with explicit
// left/top/width/height. Dragging one of its 4 edges resizes a single
// dimension; dragging one of its 4 corners resizes width and height at once
// (diagonally). All handles sit fully inside the box's own border (see
// .panel-resize-* in style.css) so they can never force a stray scrollbar.
function attachResizeHandles(box, storageKey, constraints) {
  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

  ["n", "s", "e", "w", "ne", "nw", "se", "sw"].forEach((dir) => {
    const handle = document.createElement("div");
    handle.className = `panel-resize-handle panel-resize-${dir}`;
    box.appendChild(handle);

    let dragging = false;
    let startX = 0;
    let startY = 0;
    let startRect = null;

    handle.addEventListener("mousedown", (e) => {
      dragging = true;
      handle.classList.add("dragging");
      startX = e.clientX;
      startY = e.clientY;

      // Snapshot whatever size is currently rendered (whether that came
      // from the CSS default or a previous explicit resize) as explicit
      // pixel values, so this drag's delta applies against a fixed baseline.
      const parentRect = box.parentElement.getBoundingClientRect();
      const boxRect = box.getBoundingClientRect();
      startRect = {
        left: boxRect.left - parentRect.left,
        top: boxRect.top - parentRect.top,
        width: boxRect.width,
        height: boxRect.height,
      };
      box.style.left = `${startRect.left}px`;
      box.style.top = `${startRect.top}px`;
      box.style.width = `${startRect.width}px`;
      box.style.height = `${startRect.height}px`;

      document.body.style.userSelect = "none";
      e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      let { left, top, width, height } = startRect;

      // Corner handles (e.g. "se") combine both axes -- checking each
      // character independently lets one drag resize width and height
      // together for a true diagonal resize.
      if (dir.includes("e")) {
        width = clamp(startRect.width + dx, constraints.minWidth, constraints.maxWidth);
      } else if (dir.includes("w")) {
        const right = startRect.left + startRect.width;
        width = clamp(startRect.width - dx, constraints.minWidth, constraints.maxWidth);
        left = right - width;
      }

      if (dir.includes("s")) {
        height = clamp(startRect.height + dy, constraints.minHeight, constraints.maxHeight);
      } else if (dir.includes("n")) {
        const bottom = startRect.top + startRect.height;
        height = clamp(startRect.height - dy, constraints.minHeight, constraints.maxHeight);
        top = bottom - height;
      }

      box.style.left = `${left}px`;
      box.style.top = `${top}px`;
      box.style.width = `${width}px`;
      box.style.height = `${height}px`;
    });

    window.addEventListener("mouseup", () => {
      if (!dragging) return;
      dragging = false;
      handle.classList.remove("dragging");
      document.body.style.userSelect = "";
      localStorage.setItem(storageKey, JSON.stringify({
        left: parseFloat(box.style.left),
        top: parseFloat(box.style.top),
        width: parseFloat(box.style.width),
        height: parseFloat(box.style.height),
      }));
    });
  });
}

// ---------------- Resizable panel boxes ----------------
// Each of the four main panels (Dataset / Fine-Tune / RAG / Chat) is an
// independently resizable box. Geometry is remembered per panel in
// localStorage; a panel that's never been resized just keeps the CSS
// default (100% width, 100% height -- i.e. it fills the available space).
function makeResizablePanel(panel, storageKey, constraints) {
  const stored = JSON.parse(localStorage.getItem(storageKey) || "null");
  if (stored) {
    panel.style.left = `${stored.left}px`;
    panel.style.top = `${stored.top}px`;
    panel.style.width = `${stored.width}px`;
    panel.style.height = `${stored.height}px`;
  }
  attachResizeHandles(panel, storageKey, constraints);
}

const PANEL_RESIZE_CONSTRAINTS = { minWidth: 360, maxWidth: 2000, minHeight: 240, maxHeight: 2000 };
["dataset", "finetune", "rag", "chat"].forEach((name) => {
  makeResizablePanel(
    document.getElementById(`panel-${name}`),
    `myself.panelGeom.${name}`,
    PANEL_RESIZE_CONSTRAINTS
  );
});

// ---------------- Move handle (drag a whole box to a new spot) ----------------
// Only added to inner boxes (see convertPanelInnerBoxes below), not panels.
// Grabbing the .box-move-handle grip updates left/top only -- width/height
// are left alone -- and the result is clamped so the box can't be dragged
// entirely outside its panel where it'd be unreachable. Persists to the
// same localStorage entry the resize handles use, so a box's position and
// size always travel together.
function attachMoveHandle(box, storageKey, positionClass) {
  const handle = document.createElement("div");
  handle.className = positionClass ? `box-move-handle ${positionClass}` : "box-move-handle";
  handle.title = "Drag to move";
  box.appendChild(handle);

  let dragging = false;
  let startX = 0;
  let startY = 0;
  let startLeft = 0;
  let startTop = 0;

  handle.addEventListener("mousedown", (e) => {
    dragging = true;
    handle.classList.add("dragging");
    startX = e.clientX;
    startY = e.clientY;

    const parentRect = box.parentElement.getBoundingClientRect();
    const boxRect = box.getBoundingClientRect();
    startLeft = boxRect.left - parentRect.left;
    startTop = boxRect.top - parentRect.top;
    box.style.left = `${startLeft}px`;
    box.style.top = `${startTop}px`;

    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;

    const parentRect = box.parentElement.getBoundingClientRect();
    const maxLeft = Math.max(0, parentRect.width - box.offsetWidth);
    const maxTop = Math.max(0, parentRect.height - box.offsetHeight);

    const left = Math.min(maxLeft, Math.max(0, startLeft + dx));
    const top = Math.min(maxTop, Math.max(0, startTop + dy));

    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    document.body.style.userSelect = "";
    localStorage.setItem(storageKey, JSON.stringify({
      left: parseFloat(box.style.left),
      top: parseFloat(box.style.top),
      width: parseFloat(box.style.width),
      height: parseFloat(box.style.height),
    }));
  });
}

// ---------------- Resizable inner boxes (the cards inside each panel) ----------------
// Same idea, one level deeper: every card marked .resizable-box (the
// "Chatting with" banner, model-path cards, log boxes, etc.) becomes its own
// independently resizable box, using the same 8-handle mechanism as a panel.
// A box normally sits in the panel's ordinary vertical flow, so converting
// it to an absolutely-positioned box first requires *measuring* the spot it
// would have occupied -- that measurement only works while its panel is
// actually visible (a display:none ancestor collapses everything inside it
// to a zero-size rect), so each panel's boxes are converted lazily the
// first time that panel is opened, not all at once on page load.
const INNER_BOX_CONSTRAINTS = { minWidth: 220, maxWidth: 1600, minHeight: 50, maxHeight: 1200 };

function convertPanelInnerBoxes(panel) {
  const boxes = Array.from(panel.querySelectorAll(".resizable-box")).filter(
    (box) => box.dataset.resizeReady !== "true"
  );
  if (boxes.length === 0) return;

  // Pass 1: measure every box's natural flow geometry BEFORE converting any
  // of them -- converting one early would shift where its still-in-flow
  // siblings render, corrupting their measurements. Anything that starts
  // hidden (the dataset preview card, the fine-tune "training complete"
  // card) is temporarily revealed just for this measurement.
  const geometries = boxes.map((box) => {
    const storageKey = `myself.boxGeom.${box.id}`;
    const stored = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (stored) return stored;

    const wasHiddenByClass = box.classList.contains("hidden");
    const previousInlineDisplay = box.style.display;
    if (wasHiddenByClass) box.classList.remove("hidden");
    if (previousInlineDisplay === "none") box.style.display = "block";

    const parentRect = box.parentElement.getBoundingClientRect();
    const rect = box.getBoundingClientRect();
    const geometry = {
      left: rect.left - parentRect.left,
      top: rect.top - parentRect.top,
      width: rect.width,
      height: rect.height,
    };

    if (wasHiddenByClass) box.classList.add("hidden");
    box.style.display = previousInlineDisplay;
    return geometry;
  });

  // Pass 2: apply the frozen geometry now that every box's natural position
  // has already been captured, so nothing visibly shifts.
  boxes.forEach((box, i) => {
    const parent = box.parentElement;
    if (getComputedStyle(parent).position === "static") {
      parent.style.position = "relative";
    }

    box.dataset.resizeReady = "true";
    box.style.position = "absolute";
    box.style.margin = "0";
    box.style.left = `${geometries[i].left}px`;
    box.style.top = `${geometries[i].top}px`;
    box.style.width = `${geometries[i].width}px`;
    box.style.height = `${geometries[i].height}px`;

    attachResizeHandles(box, `myself.boxGeom.${box.id}`, INNER_BOX_CONSTRAINTS);
    attachMoveHandle(box, `myself.boxGeom.${box.id}`);
    attachMoveHandle(box, `myself.boxGeom.${box.id}`, "box-move-handle-bl");
  });
}

document.querySelectorAll(".panel.active").forEach(convertPanelInnerBoxes);

// ---------------- Sidebar navigation ----------------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const panel = document.getElementById(`panel-${btn.dataset.panel}`);
    panel.classList.add("active");
    convertPanelInnerBoxes(panel);
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

  const datasetDot = document.getElementById("datasetDot");
  const datasetText = document.getElementById("datasetStatusText");
  if (settings.dataset_path) {
    datasetDot.classList.add("ok");
    datasetText.textContent = settings.dataset_path.split("/").pop();
  } else {
    datasetDot.classList.remove("ok");
    datasetText.textContent = "No dataset";
  }

  const chatModelDot = document.getElementById("chatModelDot");
  const chatModelText = document.getElementById("chatModelStatusText");
  if (settings.chat_model_path) {
    chatModelDot.classList.add("ok");
    chatModelText.textContent = settings.chat_model_path.split("/").pop();
  } else {
    chatModelDot.classList.remove("ok");
    chatModelText.textContent = "No chat model";
  }

  // ---- active-model banners: always reflect what's actually persisted
  // (not whatever a path input currently shows), so it's obvious when a
  // browsed-to model hasn't been applied yet with "Use this model".
  setActiveBanner("ftActiveDot", "ftActiveModelText", settings.finetune.model_path);
  setActiveBanner("ragActiveDot", "ragActiveModelText", settings.rag.model_path);
  setActiveBanner("chatActiveDot", "chatActiveModelText", settings.chat_model_path);

  // pre-fill path inputs and hyper-parameter fields
  if (settings.dataset_path) document.getElementById("datasetPathInput").value = settings.dataset_path;
  if (settings.finetune.model_path) document.getElementById("ftModelPathInput").value = settings.finetune.model_path;
  if (settings.rag.model_path) document.getElementById("ragModelPathInput").value = settings.rag.model_path;
  if (settings.chat_model_path) document.getElementById("chatModelPathInput").value = settings.chat_model_path;

  // dataset per training panel, falling back to the general Dataset tab's pick
  const ftDataset = settings.finetune.dataset_path || settings.dataset_path;
  const ragDataset = settings.rag.dataset_path || settings.dataset_path;
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
}

// ---------------- Native OS folder picker ----------------
// Opens the real file manager dialog on the machine running the server
// (the dashboard only ever runs locally, so this is always the same machine
// the browser is on) instead of an in-page folder listing.
function wireBrowseButton(browseBtnId, inputId) {
  document.getElementById(browseBtnId).addEventListener("click", async () => {
    const inputEl = document.getElementById(inputId);
    const res = await api.get(`/api/fs/pick_folder?path=${encodeURIComponent(inputEl.value || "")}`);
    if (res.path) inputEl.value = res.path;
  });
}

wireBrowseButton("browseDatasetBtn", "datasetPathInput");

// A dataset folder can hold Q&A files and/or raw .txt/.md files -- describe
// whatever combination was actually found (see data_manager.dataset_summary).
function formatDatasetSummary(res) {
  const parts = [];
  if (res.count) parts.push(`${res.count} Q&A pair${res.count === 1 ? "" : "s"}`);
  if (res.text_chunk_count) parts.push(`${res.text_chunk_count} text chunk${res.text_chunk_count === 1 ? "" : "s"}`);
  return parts.length ? `Loaded ${parts.join(" and ")}.` : "No Q&A pairs or text found in this folder.";
}

function renderDatasetPreview(res) {
  const qaBlocks = (res.preview || []).map((p) => `Q: ${p.question}\nA: ${p.answer}`);
  const textBlocks = (res.text_preview || []).map((t) => `[text chunk] ${t}`);
  return qaBlocks.concat(textBlocks).join("\n\n");
}

// Generic "pick a folder, POST it, show a hint" wiring
// Used by the Fine-Tune, RAG, and Chat panels, each of which picks its own
// model and dataset instead of relying on one global setting.
function wirePathSelect({ inputId, browseBtnId, selectBtnId, hintId, endpoint, onSuccess }) {
  wireBrowseButton(browseBtnId, inputId);

  document.getElementById(selectBtnId).addEventListener("click", async () => {
    const path = document.getElementById(inputId).value.trim();
    const hint = document.getElementById(hintId);
    hint.textContent = "Checking...";
    hint.className = "hint";
    const res = await api.post(endpoint, { path });
    if (res.ok) {
      hint.textContent = onSuccess ? onSuccess(res) : `Model set: ${res.model_path}`;
      hint.className = "hint success";
    } else {
      hint.textContent = res.error;
      hint.className = "hint error";
    }
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
});

wirePathSelect({
  inputId: "chatModelPathInput",
  browseBtnId: "browseChatModelBtn",
  selectBtnId: "selectChatModelBtn",
  hintId: "chatModelHint",
  endpoint: "/api/chat/select_model",
  onSuccess: (res) => {
    // Switching to a different chat model invalidates the running
    // conversation history (it was built around whatever model answered
    // those turns), so start fresh rather than silently feeding old context
    // to a new model.
    clearChat();
    return res.rag_detected
      ? `Model set: ${res.model_path} — RAG index found, will use retrieval.`
      : `Model set: ${res.model_path} — no RAG index found, will generate directly.`;
  },
});

// Dataset selection
document.getElementById("selectDatasetBtn").addEventListener("click", async () => {
  const path = document.getElementById("datasetPathInput").value.trim();
  const hint = document.getElementById("datasetHint");
  hint.textContent = "Checking...";
  hint.className = "hint";
  const res = await api.post("/api/dataset/select", { path });
  if (res.ok) {
    hint.textContent = formatDatasetSummary(res);
    hint.className = "hint success";
    const previewCard = document.getElementById("datasetPreviewCard");
    const preview = document.getElementById("datasetPreview");
    previewCard.style.display = "block";
    preview.textContent = renderDatasetPreview(res);
    await loadDatasetFileOptions();
  } else {
    hint.textContent = res.error;
    hint.className = "hint error";
  }
  refreshStatus();
});

// Dataset: paste-to-build Q&A pairs
async function loadDatasetFileOptions() {
  const data = await api.get("/api/dataset/files");
  const select = document.getElementById("datasetTargetFile");
  select.innerHTML = "";

  data.files.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  });

  const newOpt = document.createElement("option");
  newOpt.value = "__new__";
  newOpt.textContent = "+ New file...";
  select.appendChild(newOpt);

  if (data.active_file && data.files.includes(data.active_file)) {
    select.value = data.active_file;
  } else if (data.files.length) {
    select.value = data.files[0];
  } else {
    select.value = "__new__";
  }
  toggleNewFilenameInput();
}

function toggleNewFilenameInput() {
  const select = document.getElementById("datasetTargetFile");
  const newInput = document.getElementById("datasetNewFilename");
  newInput.classList.toggle("hidden", select.value !== "__new__");
}

document.getElementById("datasetTargetFile").addEventListener("change", toggleNewFilenameInput);

document.getElementById("datasetAddBtn").addEventListener("click", async () => {
  const questionEl = document.getElementById("datasetQuestion");
  const answerEl = document.getElementById("datasetAnswer");
  const select = document.getElementById("datasetTargetFile");
  const hint = document.getElementById("datasetAddHint");

  const question = questionEl.value.trim();
  const answer = answerEl.value.trim();
  const isNew = select.value === "__new__";
  const targetFile = isNew ? document.getElementById("datasetNewFilename").value.trim() : select.value;

  if (!question || !answer) {
    hint.textContent = "Both question and answer are required.";
    hint.className = "hint error";
    return;
  }
  if (!targetFile) {
    hint.textContent = "Enter a filename for the new dataset file.";
    hint.className = "hint error";
    return;
  }

  hint.textContent = "Saving...";
  hint.className = "hint";
  const res = await api.post("/api/dataset/add", {
    question,
    answer,
    target_file: targetFile,
    create_new: isNew,
  });

  if (res.ok) {
    hint.textContent = `Saved to ${res.saved_to}. Dataset now has ${res.count} pairs.`;
    hint.className = "hint success";
    questionEl.value = "";
    answerEl.value = "";
    const previewCard = document.getElementById("datasetPreviewCard");
    const preview = document.getElementById("datasetPreview");
    previewCard.style.display = "block";
    preview.textContent = renderDatasetPreview(res);
    await loadDatasetFileOptions();
  } else {
    hint.textContent = res.error;
    hint.className = "hint error";
  }
});

// Fine-tune / RAG training
let pollTimer = null;

function startPolling(logElId, onDone) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const status = await api.get("/api/train/status");
    const logEl = document.getElementById(logElId);
    logEl.textContent = status.logs.length ? status.logs.join("\n") : "Idle.";
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
  logEl.textContent = "Starting...";
  startPolling("finetuneLog", (status) => {
    if (status.done && !status.error) {
      document.getElementById("ftCompleteCard").classList.remove("hidden");
    }
  });
});

async function finalizeFinetune(body) {
  const hint = document.getElementById("ftFinalizeHint");
  hint.textContent = "Saving...";
  hint.className = "hint";
  const res = await api.post("/api/train/finetune/finalize", body);
  if (res.ok) {
    hint.textContent = `Saved to ${res.saved_to}.`;
    hint.className = "hint success";
    document.getElementById("ftCompleteCard").classList.add("hidden");
    refreshStatus();
  } else {
    hint.textContent = res.error;
    hint.className = "hint error";
  }
}

document.getElementById("ftSaveHereBtn").addEventListener("click", () => {
  const destination = document.getElementById("ftSaveDestInput").value.trim();
  if (!destination) {
    const hint = document.getElementById("ftFinalizeHint");
    hint.textContent = "Enter a destination path.";
    hint.className = "hint error";
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
  logEl.textContent = "Starting...";
  startPolling("ragLog");
});

// Chat: compute device
document.getElementById("chatDevice").addEventListener("change", async () => {
  const device = document.getElementById("chatDevice").value;
  const hint = document.getElementById("chatDeviceHint");
  hint.textContent = "Saving...";
  hint.className = "hint";
  const res = await api.post("/api/chat/device", { device });
  if (res.ok) {
    hint.textContent = `Chat will use: ${res.resolved}`;
    hint.className = "hint success";
  } else {
    hint.textContent = res.error;
    hint.className = "hint error";
  }
});

// Chat: conversation memory + response length
async function saveChatMemorySetting(field, value) {
  const hint = document.getElementById("chatMemoryHint");
  hint.textContent = "Saving...";
  hint.className = "hint";
  const res = await api.post("/api/chat/memory", { [field]: value });
  if (res.ok) {
    hint.textContent = `Memory: last ${res.history_turns} turn${res.history_turns === 1 ? "" : "s"}, up to ${res.max_new_tokens} tokens per reply.`;
    hint.className = "hint success";
  } else {
    hint.textContent = res.error;
    hint.className = "hint error";
  }
}

document.getElementById("chatHistoryTurns").addEventListener("change", (e) => {
  saveChatMemorySetting("history_turns", Number(e.target.value));
});

document.getElementById("chatMaxTokens").addEventListener("change", (e) => {
  saveChatMemorySetting("max_new_tokens", Number(e.target.value));
});

// Chat
// Completed {question, answer} turns for the current browser session, sent
// with every request so the model can see prior turns (see model_manager.
// generate's `history` param) -- this is what lets "continue that" or
// "write the rest" work instead of each message starting from scratch.
let chatHistory = [];

function clearChat() {
  chatHistory = [];
  document.getElementById("chatLog").innerHTML = "";
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

// Bot replies may contain Markdown, raw HTML, and/or LaTeX ($...$, $$...$$).
// Parse Markdown, sanitize the resulting HTML (the model's output is
// untrusted input), then let KaTeX's auto-render find and typeset any math.
function renderBotContent(bubble, text) {
  const textDiv = bubble.querySelector(".bubble-text");
  textDiv.innerHTML = DOMPurify.sanitize(marked.parse(text));
  if (window.renderMathInElement) {
    renderMathInElement(textDiv, { throwOnError: false });
  }
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

// Only one question can be in flight at a time -- the backend also refuses
// a second /api/chat while one is running, but guarding here means the
// Send button can never even fire the extra request in the first place.
let chatBusy = false;

function setChatBusy(busy) {
  chatBusy = busy;
  document.getElementById("chatSendBtn").disabled = busy;
  document.getElementById("chatStopBtn").disabled = !busy;
}

async function sendChat() {
  if (chatBusy) return;
  const input = document.getElementById("chatInput");
  const question = input.value.trim();
  if (!question) return;
  appendBubble(question, "user");
  input.value = "";
  autoGrowChatInput();

  const thinkingBubble = appendBubble("Thinking...", "bot");
  const chatLog = document.getElementById("chatLog");

  setChatBusy(true);
  try {
    const res = await api.post("/api/chat", { question, history: chatHistory });
    if (res.ok) {
      renderBotContent(thinkingBubble, res.answer);
      addCopyButton(thinkingBubble, res.answer);
      chatHistory.push({ question, answer: res.answer });
      if (res.stopped) {
        const note = document.createElement("span");
        note.className = "stopped-note";
        note.textContent = "Generation stopped early.";
        thinkingBubble.appendChild(note);
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
  await api.post("/api/chat/stop", {});
  stopBtn.textContent = "Stop";
  // Left disabled: setChatBusy(false) re-enables the pair once the in-flight
  // /api/chat request actually returns with whatever it generated so far.
}

document.getElementById("chatSendBtn").addEventListener("click", sendChat);
document.getElementById("chatStopBtn").addEventListener("click", stopChat);
document.getElementById("clearChatBtn").addEventListener("click", clearChat);

document.getElementById("chatInput").addEventListener("input", autoGrowChatInput);
document.getElementById("chatInput").addEventListener("keydown", (e) => {
  // Plain Enter sends; Shift+Enter (or Ctrl/Cmd+Enter) inserts a real
  // newline, which a single-line <input> could never do -- that's why this
  // is a <textarea> now.
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});

// Init
refreshStatus();
loadDatasetFileOptions();
