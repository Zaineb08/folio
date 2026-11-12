"use strict";
/* Folio's interface. Article text and model output are untrusted, so everything
   is HTML-escaped before it reaches the page, and the server sends a strict CSP. */

const $ = (id) => document.getElementById(id);
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

const state = {
  view: "ask",
  conversationId: null,
  mode: "answer",
  scope: { collections: [], tags: [], year_min: null, year_max: null, doc_keys: [] },
  docTitles: {},
  library: null,
  status: null,
  busy: false,
  abort: null,
  viewer: null,
  lastSyncPhase: "idle",
};
const messageSources = new WeakMap();

// ---------------------------------------------------------------------- helpers

async function api(path, options = {}) {
  const init = { ...options };
  if (options.body !== undefined) {
    init.method = options.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { detail = (await response.json()).detail || detail; } catch { /* Not JSON. */ }
    throw new Error(detail);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("json") ? response.json() : response.text();
}

let toastTimer;
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4200);
}

async function copyText(text, done) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Some browsers refuse the modern call, so try the older way before giving up.
    const box = document.createElement("textarea");
    box.value = text;
    document.body.append(box);
    box.select();
    const copied = document.execCommand("copy");
    box.remove();
    if (!copied) return toast("Your browser blocked copying to the clipboard.");
  }
  toast(done);
}

const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

// -------------------------------------------------------------------- markdown

const FLAG_OPEN = "", FLAG_MID = "", FLAG_CLOSE = "";

function citeHtml(n, ctx) {
  const source = ctx.sources?.find((s) => s.n === n);
  if (!source) {
    if (ctx.streaming) return "";
    return `<span class="cite bad" title="This answer cited a source that doesn't exist">${n}</span>`;
  }
  const where = source.page ? `, p. ${source.page}` : "";
  return `<button type="button" class="cite" data-n="${n}" title="${esc(source.cite + where)}">${n}</button>`;
}

function inline(raw, ctx) {
  let out = esc(raw);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*\w])\*([^*\n]+)\*(?![\w*])/g, "$1<em>$2</em>");
  out = out.replace(/\s*\[(\d+(?:\s*[,;]\s*\d+)*)\]/g, (_, group) =>
    group.split(/[,;]/).map((n) => citeHtml(Number(n.trim()), ctx)).join(""));
  out = out.replace(new RegExp(`${FLAG_OPEN}(\\d+)${FLAG_MID}`, "g"), (_, i) =>
    `<mark class="verify" title="${esc(ctx.flags?.[Number(i)]?.detail || "Check this against the source.")}">`);
  return out.replaceAll(FLAG_CLOSE, "</mark>");
}

function markFlags(text, flags) {
  let out = text;
  (flags || []).forEach((flag, index) => {
    if (flag.text.startsWith("|")) return; // A table row can't be wrapped in a <mark>.
    const at = out.indexOf(flag.text);
    if (at >= 0) {
      out = `${out.slice(0, at)}${FLAG_OPEN}${index}${FLAG_MID}${flag.text}${FLAG_CLOSE}${out.slice(at + flag.text.length)}`;
    }
  });
  return out;
}

function renderMarkdown(text, ctx) {
  const clean = text.replace(/[-]/g, "");
  const lines = markFlags(clean, ctx.flags).split(/\r?\n/);
  const html = [];
  let i = 0;
  const isRule = (line) => /^\s*\|?[\s:|-]+\|?\s*$/.test(line) && line.includes("-");
  const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${inline(heading[2], ctx)}</h${level}>`);
      i += 1;
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && isRule(lines[i + 1])) {
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows.push(cells(lines[i])); i += 1; }
      html.push(
        `<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${inline(c, ctx)}</th>`).join("")}</tr></thead><tbody>` +
        rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c, ctx)}</td>`).join("")}</tr>`).join("") +
        "</tbody></table></div>");
      continue;
    }
    const listMatch = line.match(/^\s*([-*+]|\d+[.)])\s+/);
    if (listMatch) {
      const ordered = /\d/.test(listMatch[1]);
      const items = [];
      while (i < lines.length) {
        const match = lines[i].match(/^\s*([-*+]|\d+[.)])\s+(.*)$/);
        if (match) { items.push(match[2]); i += 1; }
        else if (lines[i].trim() && /^\s+/.test(lines[i])) { items[items.length - 1] += ` ${lines[i].trim()}`; i += 1; }
        else if (!lines[i].trim() && lines[i + 1]?.match(/^\s*([-*+]|\d+[.)])\s+/)) { i += 1; }
        else break;
      }
      const tag = ordered ? "ol" : "ul";
      html.push(`<${tag}>${items.map((item) => `<li>${inline(item, ctx)}</li>`).join("")}</${tag}>`);
      continue;
    }
    if (line.startsWith(">")) {
      const quote = [];
      while (i < lines.length && lines[i].startsWith(">")) { quote.push(lines[i].replace(/^>\s?/, "")); i += 1; }
      html.push(`<blockquote>${inline(quote.join(" "), ctx)}</blockquote>`);
      continue;
    }
    const paragraph = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|>|\s*([-*+]|\d+[.)])\s)/.test(lines[i])
           && !(lines[i].includes("|") && isRule(lines[i + 1] || ""))) {
      paragraph.push(lines[i]);
      i += 1;
    }
    if (!paragraph.length) { paragraph.push(line); i += 1; }
    html.push(`<p>${inline(paragraph.join(" "), ctx)}</p>`);
  }
  return html.join("");
}

// ----------------------------------------------------------------------- views

function showView(view) {
  state.view = view;
  for (const name of ["ask", "sources", "matrix", "library", "discover"]) $(`view-${name}`).hidden = name !== view;
  document.querySelectorAll(".nav-item").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  if (view === "library") refreshLibrary();
  if (view === "discover") $("disc-q").focus();
  if (view === "sources") $("claim-text").focus();
  if (view === "matrix") refreshMatrix().catch((error) => toast(error.message));
  if (view === "ask") $("question").focus();
}

// ---------------------------------------------------------------------- status

async function refreshStatus() {
  let status;
  try {
    status = await api("/api/status");
  } catch {
    $("st-zotero").textContent = "Folio stopped";
    $("dot-zotero").className = "dot bad";
    setTimeout(refreshStatus, 5000);
    return;
  }
  state.status = status;

  const z = status.zotero;
  $("dot-zotero").className = `dot ${z.found ? "ok" : "warn"}`;
  $("st-zotero").textContent = !z.installed && !z.found ? "Not installed" : !z.found ? "Open it once" : z.running ? "Connected" : "Library found";

  $("dot-llm").className = `dot ${status.llm.up && status.llm.installed ? "ok" : "bad"}`;
  renderModelMenu(status);

  const count = status.library.docs;
  $("nav-count").textContent = count ? count.toLocaleString() : "";

  const sync = status.sync;
  const box = $("sync"), bar = $("sync-bar"), label = $("sync-label");
  box.hidden = sync.phase === "idle";
  if (sync.phase === "indexing" && sync.total) {
    label.textContent = `Indexing ${sync.done} of ${sync.total}`;
    label.title = sync.current || "";
    bar.classList.remove("busy");
    bar.style.width = `${Math.round((sync.done / sync.total) * 100)}%`;
  } else if (sync.phase !== "idle") {
    label.textContent = sync.phase === "downloading" ? "Downloading search models (first run only)" : "Reading Zotero";
    bar.classList.add("busy");
    bar.style.width = "";
  }
  if (state.lastSyncPhase !== "idle" && sync.phase === "idle") {
    state.library = null;
    if (sync.error) toast(sync.error);
    if (state.view === "library") refreshLibrary();
    renderWelcome();
  }
  state.lastSyncPhase = sync.phase;
  renderWelcome();
  setTimeout(refreshStatus, sync.phase === "idle" && !status.download.active ? 8000 : 1200);
}

// ---------------------------------------------------------------- answer model

function isClaude(model) {
  return !!state.status?.claude.models.some((m) => m.id === model);
}

// Folio's own models that aren't downloaded yet, offered in the menu with their size.
function missingModel(model) {
  const status = state.status;
  if (!status || status.llm.models.includes(model)) return null;
  return status.suggested.find((m) => m.id === model) || null;
}

function localLabel(model, status) {
  const info = status.suggested.find((m) => m.id === model);
  const fits = info?.suits ? ", suits this PC" : "";
  if (!status.llm.models.includes(model) && info) return `${model} (download ${info.download_gb} GB${fits})`;
  return `${model}${fits}`;
}

function renderModelMenu(status) {
  const select = $("model-select");
  const selectedIsClaude = status.claude.models.some((m) => m.id === status.llm.selected);
  const local = [...new Set([...status.llm.models, ...status.suggested.map((m) => m.id)])];
  if (!selectedIsClaude && !local.includes(status.llm.selected)) local.unshift(status.llm.selected);
  const signature = JSON.stringify([local, status.llm.models, status.suggested, status.claude.models, status.claude.has_key]);
  if (select.dataset.list !== signature) {
    select.dataset.list = signature;
    const option = (value, label) => `<option value="${esc(value)}">${esc(label)}</option>`;
    select.innerHTML =
      `<optgroup label="On this PC">${local.map((m) => option(m, localLabel(m, status))).join("")}</optgroup>` +
      `<optgroup label="Claude: passages go to Anthropic">${status.claude.models
        .map((m) => option(m.id, status.claude.has_key ? m.label : `${m.label} (add key)`)).join("")}</optgroup>`;
  }
  if (document.activeElement !== select) select.value = status.llm.selected;
  $("model-note").hidden = !selectedIsClaude;
  $("manage-key").textContent = status.claude.has_key ? "Manage Claude API key" : "Add Claude API key";

  // The selected model isn't downloaded yet, for example on a first start: offer it.
  const download = status.download;
  const missing = missingModel(status.llm.selected);
  const get = $("model-get");
  get.hidden = !missing || download.active;
  if (missing) get.textContent = `Download ${missing.id} (${missing.download_gb} GB)`;

  const box = $("model-download");
  box.hidden = !download.active;
  if (download.active) {
    const percent = download.total ? Math.round((download.done / download.total) * 100) : 0;
    $("model-download-label").textContent = download.total ? `Downloading ${download.model}: ${percent}%` : `Starting the download of ${download.model}`;
    $("model-download-bar").style.width = `${percent}%`;
  }
  if (state.downloading && !download.active) {
    toast(download.error || `${download.model} is downloaded and ready.`);
  }
  state.downloading = download.active;
}

async function downloadModel(model) {
  const info = missingModel(model);
  if (!info) return;
  if (!confirm(`Download ${model}? It's ${info.download_gb} GB, and Folio switches to it once it's done. You can keep using Folio meanwhile.`)) return;
  try {
    state.status.download = await api("/api/models/download", { body: { model, select: true } });
    renderModelMenu(state.status);
    setTimeout(refreshStatus, 1000);
  } catch (error) {
    toast(error.message);
  }
}

let keyDialogModel = null;

function openKeyDialog(model = null) {
  keyDialogModel = model;
  $("key-input").value = "";
  $("key-error").hidden = true;
  $("key-remove").hidden = !state.status?.claude.has_key;
  $("key-dialog").showModal();
  $("key-input").focus();
}

async function saveKey() {
  const key = $("key-input").value.trim();
  const error = $("key-error");
  if (!key) return;
  const button = $("key-save");
  button.disabled = true;
  button.textContent = "Checking key";
  error.hidden = true;
  try {
    await api("/api/claude/key", { body: { key } });
    $("key-input").value = "";
    state.status.claude.has_key = true;
    $("key-dialog").close();
    if (keyDialogModel) await chooseModel(keyDialogModel);
    renderModelMenu(state.status);
    toast("Key saved. Claude is ready in the model menu.");
  } catch (failure) {
    error.textContent = failure.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Save key";
  }
}

async function removeKey() {
  await api("/api/claude/key", { method: "DELETE" });
  $("key-dialog").close();
  const status = await api("/api/status");
  state.status = status;
  renderModelMenu(status);
  toast("Key removed. Folio is back on the model on this PC.");
}

async function chooseModel(model) {
  await api("/api/settings", { body: { llm_model: model } });
  state.status.llm.selected = model;
  $("model-select").value = model;
  renderModelMenu(state.status);
}

function formatCost(cost) {
  return `$${cost < 0.01 ? cost.toFixed(3) : cost.toFixed(2)}`;
}

function metaLine(meta) {
  if (!meta) return "";
  const parts = [];
  if (meta.provider === "claude") {
    parts.push(meta.fell_back ? `${meta.label} declined, so ${meta.served_label} wrote this` : `Written by ${meta.served_label}`);
    if (meta.cost_usd !== null && meta.cost_usd !== undefined) parts.push(`about ${formatCost(meta.cost_usd)}`);
    if (meta.checked_by && meta.checked_by !== meta.model) parts.push(`checked on this PC by ${meta.checked_by}`);
    if (meta.truncated) parts.push("cut off at the length limit");
  } else {
    parts.push(`Written on this PC by ${meta.model}`);
  }
  if (meta.coverage) parts.push(`based on ${meta.coverage.used} of ${meta.coverage.total} passages from this paper`);
  return `<p class="msg-meta">${esc(parts.join(" · "))}</p>`;
}

const EVIDENCE_NOTES = {
  weak: "Your papers only touch on this. Their passages match the question poorly, so treat this answer with caution and check the sources.",
  sample: "This reads a sample of your library, with a passage or two from each article, not whole articles.",
};

function setEvidence(el, level) {
  const note = el.querySelector(".evidence-note");
  note.hidden = !EVIDENCE_NOTES[level];
  note.textContent = EVIDENCE_NOTES[level] || "";
}

// --------------------------------------------------------------------- welcome

function renderWelcome() {
  const el = $("welcome");
  const hasMessages = $("messages").children.length > 0;
  el.hidden = hasMessages;
  if (hasMessages || !state.status) return;
  const { library, zotero, llm, sync } = state.status;
  const signature = JSON.stringify([library.chunks > 0, zotero.found, llm.installed, sync.phase]);
  if (el.dataset.sig === signature) return;
  el.dataset.sig = signature;

  if (library.chunks > 0) {
    const starters = [
      "What are the main findings across my library?",
      "Which methods do these articles use, and how do they differ?",
      "Where do these authors disagree?",
      "What do these articles say is still unknown?",
    ];
    el.innerHTML = `<h1>What are you trying to find out?</h1>
      <p>Answers come only from your articles. Every claim links to the page it came from, and claims that don't match their source are flagged.</p>
      <ul class="starters">${starters.map((s) => `<li><button type="button" class="starter">${esc(s)}</button></li>`).join("")}</ul>`;
    return;
  }
  const indexing = sync.phase !== "idle";
  el.innerHTML = `<h1>${indexing ? "Getting your library ready" : "Let's fill your library"}</h1>
    <p>${indexing ? "Folio is indexing your articles. You can ask questions as soon as it finishes." : "Folio reads the articles in your Zotero library. Yours is empty, so here's how to start."}</p>
    ${indexing ? "" : `<ol class="setup-steps">
      <li><span>Find open-access articles under Discover and add them with one click, or drag PDFs into Zotero.</span></li>
      <li><span>Folio notices new articles by itself and indexes them on your GPU.</span></li>
      <li><span>Come back here and ask questions.</span></li>
    </ol>
    <div class="welcome-actions">
      <button type="button" class="primary-btn" data-go="discover">Discover articles</button>
      <button type="button" class="ghost-btn" data-open-zotero>Open Zotero</button>
    </div>`}`;
}

// --------------------------------------------------------------- conversations

async function refreshConversations() {
  const list = await api("/api/conversations");
  $("conv-list").innerHTML = list.map((c) => `
    <li class="${c.id === state.conversationId ? "active" : ""}">
      <button type="button" class="conv-open" data-conv="${c.id}" title="${esc(c.title)}">${esc(c.title)}</button>
      <button type="button" class="icon-btn conv-del" data-del="${c.id}" title="Delete" aria-label="Delete conversation"><svg width="12" height="12"><use href="#i-x"/></svg></button>
    </li>`).join("");
}

function newConversation() {
  if (state.busy) stopAnswer();
  state.conversationId = null;
  $("messages").innerHTML = "";
  $("welcome").dataset.sig = "";
  renderWelcome();
  refreshConversations();
  showView("ask");
}

async function openConversation(id) {
  if (state.busy) stopAnswer();
  const { messages } = await api(`/api/conversations/${id}`);
  state.conversationId = id;
  const container = $("messages");
  container.innerHTML = "";
  for (const message of messages) {
    if (message.role === "user") addUserMessage(message.content);
    else {
      const el = addAssistantMessage();
      finishAssistant(el, message.content, message.sources, message.checks, false, message.meta);
    }
  }
  renderWelcome();
  refreshConversations();
  showView("ask");
  $("thread").scrollTop = $("thread").scrollHeight;
}

// -------------------------------------------------------------------- messages

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg msg-user";
  el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  $("messages").append(el);
}

function addAssistantMessage() {
  const el = document.createElement("div");
  el.className = "msg msg-assistant";
  el.innerHTML = `<div class="msg-status" hidden><span class="pulse"></span><span class="msg-status-text"></span></div>
    <p class="evidence-note" hidden></p><div class="answer"></div><div class="msg-foot"></div>`;
  $("messages").append(el);
  return el;
}

function setStatus(el, text) {
  const box = el.querySelector(".msg-status");
  box.hidden = !text;
  if (text) box.querySelector(".msg-status-text").textContent = text;
}

function sourceRow(source) {
  const page = source.page
    ? (source.page_end && source.page_end !== source.page ? `pp. ${source.page}–${source.page_end}` : `p. ${source.page}`)
    : "abstract";
  return `<li><button type="button" class="source" data-n="${source.n}">
      <span class="cite">${source.n}</span>
      <span><span class="source-cite">${esc(source.cite)}</span><span class="source-title">${esc(source.title)}</span></span>
      <span class="source-page">${page}</span>
    </button></li>`;
}

function finishAssistant(el, text, sources = [], checks = null, stopped = false, meta = null) {
  messageSources.set(el, sources);
  setStatus(el, "");
  setEvidence(el, meta?.evidence);
  el.querySelector(".answer").innerHTML = renderMarkdown(text, { sources, flags: checks?.flags });
  const foot = el.querySelector(".msg-foot");
  if (!sources.length) { foot.innerHTML = ""; return; }

  const citedNumbers = new Set(checks?.cited ?? [...text.matchAll(/\[(\d+)/g)].map((m) => Number(m[1])));
  const cited = sources.filter((s) => citedNumbers.has(s.n));
  const others = sources.filter((s) => !citedNumbers.has(s.n));

  let checkHtml = "";
  if (stopped) {
    checkHtml = `<div class="check-line">Stopped early, so this answer wasn't checked.</div>`;
  } else if (checks && checks.claims) {
    const good = checks.claims - checks.flagged;
    const flagged = checks.flagged;
    const summary = flagged
      ? `<span class="warnish">${plural(flagged, "claim")} to verify</span> <span>The other ${good} of ${checks.claims} passed the check against their cited passages.</span>
         <button type="button" class="check-toggle">Show which</button>`
      : `<span class="good">${checks.claims} of ${checks.claims} claims</span> <span>passed the check against their cited passages. The check can miss things, so read the sources for anything you'll rely on.</span>`;
    const details = flagged
      ? `<ul class="check-list" hidden>${checks.flags.map((f) => `<li><q>${esc(f.text.replace(/\s*\[\d+(?:\s*[,;]\s*\d+)*\]/g, ""))}</q><small>${esc(f.detail)}</small></li>`).join("")}</ul>`
      : "";
    checkHtml = `<div class="check-line">${summary}</div>${details}`;
  }

  foot.innerHTML = `${metaLine(meta)}${checkHtml}
    <div class="msg-actions">
      <button type="button" class="link-btn" data-act="copy">Copy with references</button>
      <button type="button" class="link-btn" data-act="bibtex">Copy BibTeX</button>
    </div>
    ${cited.length ? `<p class="sources-title">Cited</p><ul class="sources">${cited.map(sourceRow).join("")}</ul>` : ""}
    ${others.length ? `<details class="more-sources"><summary>${plural(others.length, "more passage")} searched, not cited</summary><ul class="sources">${others.map(sourceRow).join("")}</ul></details>` : ""}`;
  el.dataset.answer = text;
}

function answerAsThesisText(text, sources) {
  const byN = Object.fromEntries(sources.map((s) => [s.n, s]));
  const used = new Set();
  const body = text.replace(/\s*\[(\d+(?:\s*[,;]\s*\d+)*)\](?:\s*\[(\d+(?:\s*[,;]\s*\d+)*)\])*/g, (match) => {
    const numbers = [...match.matchAll(/\d+/g)].map((m) => Number(m[0]));
    const parts = numbers.filter((n) => byN[n]).map((n) => {
      used.add(n);
      const source = byN[n];
      return source.page ? `${source.cite}, p. ${source.page}` : source.cite;
    });
    return parts.length ? ` (${[...new Set(parts)].join("; ")})` : "";
  });
  const references = [...new Set([...used].map((n) => byN[n].apa))].sort();
  return references.length ? `${body}\n\nReferences\n\n${references.join("\n\n")}` : body;
}

// --------------------------------------------------------------------- asking

function setBusy(busy) {
  state.busy = busy;
  const send = $("send");
  send.title = busy ? "Stop" : "Send";
  send.setAttribute("aria-label", send.title);
  send.querySelector("use").setAttribute("href", busy ? "#i-stop" : "#i-send");
  updateSend();
}

function updateSend() {
  $("send").disabled = !state.busy && !$("question").value.trim();
}

function stopAnswer() {
  api("/api/ask/stop", { method: "POST" }).catch(() => {});
}

async function ask(question, mode = state.mode) {
  if (state.busy || !question.trim()) return;
  showView("ask");
  addUserMessage(question);
  renderWelcome();
  const el = addAssistantMessage();
  setStatus(el, "Searching your library");
  $("question").value = "";
  autoGrow();
  setBusy(true);
  scrollThread(true);

  let text = "", sources = [], checks = null, stopped = false, frame = 0, meta = null;
  const repaint = () => {
    frame = 0;
    el.querySelector(".answer").innerHTML = renderMarkdown(text, { sources, streaming: true });
    scrollThread();
  };
  const handlers = {
    conversation: (e) => { state.conversationId = e.id; refreshConversations(); },
    status: (e) => setStatus(el, e.text),
    sources: (e) => { sources = e.sources; messageSources.set(el, sources); setEvidence(el, e.evidence); },
    token: (e) => { text += e.text; if (!frame) frame = requestAnimationFrame(repaint); },
    check: (e) => { checks = e; },
    meta: (e) => { meta = e; },
    reset: () => {
      text = "";
      if (frame) { cancelAnimationFrame(frame); frame = 0; }
      el.querySelector(".answer").innerHTML = "";
    },
    done: (e) => { stopped = !!e.stopped; },
    error: (e) => {
      setStatus(el, "");
      const box = document.createElement("div");
      box.className = "err";
      box.textContent = e.message;
      el.querySelector(".msg-foot").append(box);
      el.dataset.failed = "1";
    },
  };

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: state.conversationId, mode, scope: state.scope }),
    });
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try { detail = (await response.json()).detail || detail; } catch { /* Not JSON. */ }
      throw new Error(detail);
    }
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        if (chunk.startsWith("data: ")) {
          const event = JSON.parse(chunk.slice(6));
          handlers[event.type]?.(event);
        }
      }
    }
    if (frame) cancelAnimationFrame(frame);
    if (!el.dataset.failed) finishAssistant(el, text, sources, checks, stopped, meta);
    else if (text) el.querySelector(".answer").innerHTML = renderMarkdown(text, { sources });
  } catch (error) {
    handlers.error({ message: error.message || "Lost the connection to Folio." });
  } finally {
    setStatus(el, "");
    setBusy(false);
    refreshConversations();
    $("question").focus();
  }
}

function scrollThread(force = false) {
  const thread = $("thread");
  const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 140;
  if (force || nearBottom) thread.scrollTop = thread.scrollHeight;
}

function autoGrow() {
  const box = $("question");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 200)}px`;
  updateSend();
}

// ----------------------------------------------------------------------- scope

async function libraryData() {
  if (!state.library) state.library = await api("/api/library");
  return state.library;
}

function scopeCount() {
  const s = state.scope;
  return s.collections.length + s.tags.length + s.doc_keys.length + (s.year_min || s.year_max ? 1 : 0);
}

function renderScopeChips() {
  const s = state.scope;
  const chips = [
    ...s.doc_keys.map((k) => ["doc_keys", k, state.docTitles[k] || "One article"]),
    ...s.collections.map((c) => ["collections", c, c]),
    ...s.tags.map((t) => ["tags", t, `#${t}`]),
  ];
  if (s.year_min || s.year_max) chips.push(["years", "", `${s.year_min || "…"}–${s.year_max || "…"}`]);
  $("scope-chips").innerHTML = chips.map(([kind, value, label]) => `
    <span class="chip"><span title="${esc(label)}">${esc(label)}</span>
      <button type="button" data-unscope="${esc(kind)}" data-value="${esc(value)}" aria-label="Remove filter"><svg width="10" height="10"><use href="#i-x"/></svg></button></span>`).join("");
  $("scope-label").textContent = scopeCount() ? plural(scopeCount(), "filter") : "Whole library";
}

async function renderScopePopover() {
  const lib = await libraryData();
  const s = state.scope;
  const group = (title, kind, entries) => `<div class="pop-group"><h2>${title}</h2>${entries.length
    ? entries.map((e) => `<label><input type="checkbox" data-kind="${kind}" value="${esc(e.name)}" ${s[kind].includes(e.name) ? "checked" : ""}> <span>${esc(e.name)} (${e.count})</span></label>`).join("")
    : `<p class="pop-empty">None yet</p>`}</div>`;
  $("scope-pop").innerHTML =
    group("Collections", "collections", lib.collections) +
    group("Tags", "tags", lib.tags) +
    `<div class="pop-group"><h2>Published</h2><div class="pop-years">
       <input type="number" id="scope-from" aria-label="From year" placeholder="From" value="${s.year_min ?? ""}">
       <input type="number" id="scope-to" aria-label="To year" placeholder="To" value="${s.year_max ?? ""}">
     </div></div>`;
}

// --------------------------------------------------------------------- library

const STATUS_BADGES = {
  ok: ["ok", "Indexed"],
  abstract_only: ["warn", "Abstract only"],
  metadata_only: ["warn", "No text"],
  scanned: ["warn", "Scanned, no text"],
  encrypted: ["bad", "Password-protected"],
  error: ["bad", "Couldn't read"],
  pending: ["", "Waiting"],
};

async function refreshLibrary() {
  state.library = null;
  const lib = await libraryData();
  const chunks = lib.docs.reduce((total, doc) => total + doc.chunks, 0);
  $("lib-summary").textContent = lib.docs.length
    ? `${plural(lib.docs.length, "article")}, ${plural(chunks, "searchable passage")}`
    : "Nothing here yet.";
  const problems = lib.docs.filter((doc) => doc.status !== "ok" && doc.status !== "pending");
  const notice = $("lib-notice");
  notice.hidden = !problems.length;
  if (problems.length) {
    notice.textContent = `${plural(problems.length, "article")} can't be searched in full. Attach a PDF in Zotero, or for scans, run OCR on the PDF first.`;
  }
  renderLibraryList();
}

function renderLibraryList() {
  const lib = state.library;
  if (!lib) return;
  const filter = $("lib-filter").value.trim().toLowerCase();
  const docs = lib.docs.filter((doc) => !filter ||
    `${doc.title} ${doc.cite} ${doc.venue || ""} ${doc.tags.join(" ")} ${doc.collections.join(" ")}`.toLowerCase().includes(filter));
  $("lib-list").innerHTML = docs.length
    ? docs.map((doc) => {
        const [tone, label] = STATUS_BADGES[doc.status] || ["", doc.status];
        const meta = [doc.cite, doc.venue, doc.pages ? plural(doc.pages, "page") : null].filter(Boolean).join(" · ");
        return `<li class="lib-item">
          <div>
            <div class="lib-title">${esc(doc.title)}</div>
            <div class="lib-meta">${esc(meta)}</div>
          </div>
          <div class="lib-actions">
            <span class="badge ${tone}" title="${esc(doc.error || "")}">${label}</span>
            ${doc.chunks ? `<button type="button" class="link-btn" data-explain-doc="${esc(doc.key)}">Explain this paper</button>
            <button type="button" class="link-btn" data-ask-doc="${esc(doc.key)}">Ask about this</button>` : ""}
            <button type="button" class="link-btn" data-zotero-doc="${esc(doc.key)}">Show in Zotero</button>
          </div>
        </li>`;
      }).join("")
    : `<li class="empty">${lib.docs.length ? "No articles match that filter." : "Your Zotero library is empty. Add articles under Discover, or drag PDFs into Zotero."}</li>`;
}

// -------------------------------------------------------------------- discover

let discoverResults = [];

async function runDiscover(event) {
  event?.preventDefault();
  const q = $("disc-q").value.trim();
  if (!q) return;
  const list = $("disc-list");
  list.innerHTML = `<li class="empty">Searching OpenAlex</li>`;
  const params = new URLSearchParams({ q, oa: $("disc-oa").checked, sort: $("disc-sort").value });
  if ($("disc-year").value) params.set("year_min", $("disc-year").value);
  let source = "OpenAlex";
  try {
    ({ results: discoverResults, source } = await api(`/api/discover?${params}`));
  } catch (error) {
    list.innerHTML = `<li class="empty">${esc(error.message)}</li>`;
    return;
  }
  if (!discoverResults.length) {
    list.innerHTML = `<li class="empty">Nothing found. Try broader words, or include paywalled articles.</li>`;
    return;
  }
  const fallbackNote = source === "OpenAlex" ? "" :
    `<li class="disc-fallback">OpenAlex is busy, so these results come from ${esc(source)}, which has fewer open-access PDFs. Search again in a minute for fuller results.</li>`;
  list.innerHTML = fallbackNote + discoverResults.map((work, index) => {
    const authors = work.authors.slice(0, 4).map((a) => a.last).join(", ") + (work.authors.length > 4 ? " et al." : "");
    const meta = [authors, work.year, work.venue, plural(work.cited_by, "citation")].filter(Boolean).join(" · ");
    const canPdf = !!work.pdf_url;
    return `<li class="disc-item">
      <div>
        <div class="disc-title">${esc(work.title)}</div>
        <div class="disc-meta">${esc(meta)}</div>
        ${work.abstract ? `<p class="disc-abstract" data-expand title="Click to expand">${esc(work.abstract)}</p>` : ""}
      </div>
      <div class="disc-side">
        ${work.in_library
          ? `<span class="badge ok">In your library</span>`
          : `<button type="button" class="ghost-btn" data-add="${index}">Add to Zotero</button>`}
        <span class="disc-note">${canPdf ? "Open-access PDF" : work.is_oa ? "Open access, no direct PDF" : "Paywalled, details only"}</span>
      </div>
    </li>`;
  }).join("");
}

async function addDiscovered(button) {
  const work = discoverResults[Number(button.dataset.add)];
  button.disabled = true;
  button.textContent = work.pdf_url ? "Fetching PDF" : "Adding";
  try {
    const result = await api("/api/discover/add", { body: { work } });
    const note = button.parentElement.querySelector(".disc-note");
    button.outerHTML = `<span class="badge ok">Added</span>`;
    note.textContent = result.pdf_attached ? "PDF saved, indexing now" : "Details only: no PDF could be fetched";
  } catch (error) {
    button.disabled = false;
    button.textContent = "Add to Zotero";
    toast(error.message);
  }
}

// ---------------------------------------------------------------------- viewer

async function openViewer(source, trigger) {
  document.querySelectorAll(".current").forEach((el) => el.classList.remove("current"));
  trigger?.classList.add("current");
  state.viewer = { source, page: source.page || 1, rects: [] };
  $("app").classList.add("with-viewer");
  $("viewer").hidden = false;
  $("v-cite").textContent = source.page ? `${source.cite}, p. ${source.page}` : source.cite;
  $("v-title").textContent = source.title;
  $("v-title").title = source.title;

  const hasPdf = source.has_pdf && source.page;
  $("v-img").hidden = !hasPdf;
  $("v-passage").hidden = !!hasPdf;
  $("v-passage").textContent = hasPdf ? "" : source.text;
  document.querySelector(".pager").hidden = !hasPdf;

  if (hasPdf) {
    try {
      const chunk = await api(`/api/chunk/${source.chunk_id}`);
      if (state.viewer?.source === source) state.viewer.rects = chunk.rects || [];
    } catch { /* The page still shows, just without scrolling to the passage. */ }
    showViewerPage(true);
  }
}

function showViewerPage(scrollToPassage = false) {
  const viewer = state.viewer;
  if (!viewer) return;
  const { source, page } = viewer;
  const total = source.pages || page;
  $("v-page").textContent = `Page ${page} of ${total}`;
  $("v-prev").disabled = page <= 1;
  $("v-next").disabled = page >= total;
  const img = $("v-img");
  img.alt = `Page ${page} of ${source.title}`;
  img.onload = () => {
    if (!scrollToPassage) { $("v-body").scrollTop = 0; return; }
    const rects = viewer.rects.filter((rect) => rect[0] === page);
    if (rects.length && img.naturalHeight) {
      const top = Math.min(...rects.map((rect) => rect[2]));
      const ratio = img.clientHeight / (img.naturalHeight / 2); // Pages are drawn at 2x.
      $("v-body").scrollTop = Math.max(0, top * ratio - 80);
    }
  };
  img.src = `/api/page/${encodeURIComponent(source.doc_key)}/${page}?chunk=${source.chunk_id}`;
}

function closeViewer() {
  state.viewer = null;
  $("viewer").hidden = true;
  $("app").classList.remove("with-viewer");
  document.querySelectorAll(".current").forEach((el) => el.classList.remove("current"));
}

async function openInZotero(docKey, page) {
  try {
    await api("/api/zotero/open", { body: { doc_key: docKey, page } });
  } catch (error) {
    toast(error.message);
  }
}

// ---------------------------------------------------------------------- events

document.addEventListener("click", (event) => {
  const target = event.target.closest("button, [data-expand]");
  if (!target) {
    if (!event.target.closest(".scope-wrap")) closeScope();
    return;
  }
  if (!target.closest(".scope-wrap")) closeScope();

  if (target.matches(".nav-item")) return showView(target.dataset.view);
  if (target.matches(".starter")) return ask(target.textContent);
  if (target.dataset.go) return showView(target.dataset.go);
  if ("openZotero" in target.dataset) return launchZotero();
  if (target.dataset.conv) return openConversation(Number(target.dataset.conv));
  if (target.dataset.del) return deleteConversation(Number(target.dataset.del));
  if (target.dataset.add !== undefined) return addDiscovered(target);
  if ("expand" in target.dataset) return target.classList.toggle("open");
  if (target.dataset.zoteroDoc) return openInZotero(target.dataset.zoteroDoc);
  if (target.dataset.askDoc) return askAboutDoc(target.dataset.askDoc);
  if (target.dataset.explainDoc) return explainDoc(target.dataset.explainDoc);
  if (target.dataset.mxDoc) return openMatrixSource(target);
  if (target.dataset.mxRemove) return removeMatrixColumn(Number(target.dataset.mxRemove)).catch((error) => toast(error.message));
  if (target.dataset.mxIdea !== undefined) {
    const [label, question] = MX_IDEAS[Number(target.dataset.mxIdea)];
    $("mx-label").value = label;
    $("mx-question").value = question;
    return $("mx-question").focus();
  }
  if (target.dataset.claimOpen !== undefined) return openViewer(claimResults[Number(target.dataset.claimOpen)], target);
  if (target.dataset.claimCite !== undefined) return copyText(claimResults[Number(target.dataset.claimCite)].citation, "Citation copied.");
  if (target.dataset.claimRef !== undefined) return copyText(claimResults[Number(target.dataset.claimRef)].apa, "Reference copied. Check it against your style guide.");
  if ("claimDiscover" in target.dataset) {
    $("disc-q").value = $("claim-text").value.trim();
    showView("discover");
    return runDiscover();
  }
  if (target.dataset.unscope !== undefined) return removeScope(target.dataset.unscope, target.dataset.value);

  const message = target.closest(".msg-assistant");
  if (!message) return;
  const sources = messageSources.get(message) || [];
  if (target.dataset.n) {
    const source = sources.find((s) => s.n === Number(target.dataset.n));
    if (source) openViewer(source, target);
  } else if (target.matches(".check-toggle")) {
    const list = message.querySelector(".check-list");
    list.hidden = !list.hidden;
    target.textContent = list.hidden ? "Show which" : "Hide";
  } else if (target.dataset.act === "copy") {
    copyText(answerAsThesisText(message.dataset.answer, sources), "Copied, with author-year citations and references.");
  } else if (target.dataset.act === "bibtex") {
    const used = sources.filter((s) => new RegExp(`\\[[\\d,;\\s]*\\b${s.n}\\b[\\d,;\\s]*\\]`).test(message.dataset.answer));
    api("/api/cite/bibtex", { body: { doc_keys: (used.length ? used : sources).map((s) => s.doc_key) } })
      .then((text) => copyText(text, "Copied BibTeX for the cited articles."))
      .catch((error) => toast(error.message));
  }
});

async function launchZotero() {
  try {
    await api("/api/zotero/launch", { method: "POST" });
  } catch (error) {
    toast(error.message);
  }
}

async function deleteConversation(id) {
  await api(`/api/conversations/${id}`, { method: "DELETE" });
  if (state.conversationId === id) newConversation();
  else refreshConversations();
}

async function askAboutDoc(key) {
  const doc = (await libraryData()).docs.find((d) => d.key === key);
  state.docTitles[key] = doc?.cite || "One article";
  state.scope = { collections: [], tags: [], year_min: null, year_max: null, doc_keys: [key] };
  renderScopeChips();
  showView("ask");
}

async function explainDoc(key) {
  if (state.busy) return toast("Wait for the current answer to finish first.");
  const doc = (await libraryData()).docs.find((d) => d.key === key);
  if (!doc) return toast("That article is no longer in your library.");
  state.docTitles[key] = doc.cite;
  state.scope = { collections: [], tags: [], year_min: null, year_max: null, doc_keys: [key] };
  renderScopeChips();
  newConversation();
  return ask(`Explain this paper: ${doc.title}`, "explain");
}

// ---------------------------------------------------------------- find a source

let claimResults = [];
const CLAIM_GROUPS = [
  ["supports", "Supports your sentence"],
  ["disagrees", "May disagree (this label is often wrong, so read these yourself)"],
  ["related", "Related, but doesn't settle it"],
];

async function findSources(event) {
  event.preventDefault();
  const text = $("claim-text").value.trim();
  const box = $("claim-results");
  if (text.length < 10) {
    box.innerHTML = `<p class="empty">Write a full sentence, so there's a claim to look for.</p>`;
    return;
  }
  const button = $("claim-go");
  button.disabled = true;
  button.textContent = "Reading your papers";
  box.innerHTML = `<p class="claim-disclaimer">Folio reads up to 8 passages one at a time, which takes longer on a PC without a graphics card.</p>`;
  try {
    const found = await api("/api/claim", { body: { text } });
    claimResults = found.results;
    renderClaimResults(found);
  } catch (error) {
    box.innerHTML = `<p class="err">${esc(error.message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = "Find sources";
  }
}

function renderClaimResults(found) {
  const box = $("claim-results");
  const decisive = found.results.some((r) => r.verdict !== "related");
  const noneNote = decisive ? "" : `<div class="empty">
      ${found.results.length ? "Nothing in your library clearly supports or disagrees with this sentence." : "Nothing in your library looks relevant to this sentence."}
      It may need a source you don't have yet.
      <div class="welcome-actions"><button type="button" class="ghost-btn" data-claim-discover>Search Discover for it</button></div>
    </div>`;
  const groups = CLAIM_GROUPS.map(([verdict, title]) => {
    const rows = found.results.map((r, index) => [r, index]).filter(([r]) => r.verdict === verdict);
    if (!rows.length) return "";
    return `<section class="claim-group"><h2>${title} (${rows.length})</h2><ul class="claim-list">${rows.map(([r, index]) => `
      <li class="claim-item">
        <p class="claim-quote">${esc(r.quote.length > 420 ? `${r.quote.slice(0, 420)}…` : r.quote)}</p>
        <div class="claim-meta">${esc(r.citation)} · ${esc(r.title)}</div>
        ${r.note ? `<p class="claim-note">${esc(r.note)}</p>` : ""}
        <div class="claim-actions">
          <button type="button" class="link-btn" data-claim-open="${index}">See the page</button>
          <button type="button" class="link-btn" data-claim-cite="${index}">Copy citation</button>
          <button type="button" class="link-btn" data-claim-ref="${index}">Copy reference</button>
        </div>
      </li>`).join("")}</ul></section>`;
  }).join("");
  const disclaimer = found.results.length
    ? `<p class="claim-disclaimer">Sorted by ${esc(found.checked_by)}, which can be wrong. Read the passage before you cite it.</p>`
    : "";
  box.innerHTML = noneNote + disclaimer + groups;
}

// ------------------------------------------------------------ literature matrix

// Columns that suit a thesis on vulnerability and hallucination detection in AI-generated code.
const MX_IDEAS = [
  ["Languages", "Which programming languages does the study cover?"],
  ["Models evaluated", "Which AI models or code assistants did they evaluate?"],
  ["Detection approach", "How does it detect vulnerabilities or hallucinations, if it does?"],
];
const mx = { data: null, filter: "all", poll: null, wasRunning: false, drawn: new Map() };

async function refreshMatrix() {
  mx.data = await api("/api/matrix");
  renderMatrix();
  const { job } = mx.data;
  if (mx.wasRunning && !job.running) {
    if (job.error) toast(job.error);
    else if (job.stopped) toast("Stopped. The cells filled so far are kept.");
    else toast("The matrix is filled. Check a few cells against their pages.");
  }
  mx.wasRunning = job.running;
  clearTimeout(mx.poll);
  if (job.running) mx.poll = setTimeout(() => refreshMatrix().catch(() => {}), 2000);
}

function matrixRows() {
  const [kind, ...rest] = mx.filter.split(":");
  const name = rest.join(":");
  return mx.data.rows.filter((row) =>
    kind === "tag" ? row.tags.includes(name) : kind === "collection" ? row.collections.includes(name) : true);
}

function matrixCell(docKey, columnId) {
  return mx.data.cells.find((cell) => cell.doc_key === docKey && cell.column_id === columnId);
}

function renderMatrixFilter() {
  const { rows } = mx.data;
  const count = (list) => list.reduce((counts, name) => counts.set(name, (counts.get(name) || 0) + 1), new Map());
  const tags = count(rows.flatMap((row) => row.tags));
  const collections = count(rows.flatMap((row) => row.collections));
  const options = [["all", `All papers (${rows.length})`],
    ...[...collections].sort().map(([name, n]) => [`collection:${name}`, `Collection: ${name} (${n})`]),
    ...[...tags].sort().map(([name, n]) => [`tag:${name}`, `Tag: ${name} (${n})`])];
  if (!options.some(([value]) => value === mx.filter)) mx.filter = "all";
  $("mx-filter").innerHTML = options.map(([value, label]) =>
    `<option value="${esc(value)}"${value === mx.filter ? " selected" : ""}>${esc(label)}</option>`).join("");
}

function matrixCellHtml(row, column) {
  const id = `data-cell="${esc(row.key)}|${column.id}"`;
  const cell = matrixCell(row.key, column.id);
  if (!cell) return `<td ${id} class="mx-empty">Not filled yet</td>`;
  const stale = cell.stale ? `<span class="mx-stale-note">The paper changed after this was filled.</span>` : "";
  const where = (n) => cell.sources.find((s) => s.n === n);
  const pageButton = (n) => {
    const source = where(n);
    if (!source) return "";
    const label = source.page ? `p. ${source.page}` : "abstract";
    return `<button type="button" class="mx-page" data-mx-doc="${esc(row.key)}" data-mx-col="${column.id}" data-mx-n="${n}" title="See this passage">${label}</button>`;
  };
  if (!cell.text) {
    const searched = cell.sources.length ? `<span class="mx-searched">Searched ${cell.sources.map((s) => pageButton(s.n)).join("")}</span>` : "";
    return `<td ${id} class="mx-none${cell.stale ? " mx-stale" : ""}">Not stated in the passages found${searched}${stale}</td>`;
  }
  let html = esc(cell.text).replace(/\s*\[(\d+(?:\s*[,;]\s*\d+)*)\]/g, (_, group) =>
    group.split(/[,;]/).map((n) => pageButton(Number(n.trim()))).join(""));
  if (!/\[\d/.test(cell.text)) html += ` ${cell.sources.map((s) => pageButton(s.n)).join("")}`;
  if (cell.flags.length) html = `<mark class="verify" title="${esc(cell.flags.map((f) => f.detail).join(" "))}">${html}</mark>`;
  return `<td ${id} class="${cell.stale ? "mx-stale" : ""}">${html}${stale}</td>`;
}

// While the matrix fills, only cells that changed are redrawn, so clicks and hovers elsewhere aren't lost.
function renderMatrixTable(rows, columns) {
  const table = $("mx-table");
  const shape = JSON.stringify([rows.map((row) => row.key), columns.map((column) => [column.id, column.label])]);
  if (table.dataset.shape === shape) {
    for (const row of rows) {
      for (const column of columns) {
        const html = matrixCellHtml(row, column);
        const key = `${row.key}|${column.id}`;
        if (mx.drawn.get(key) === html) continue;
        const td = table.querySelector(`td[data-cell="${CSS.escape(key)}"]`);
        if (td) td.outerHTML = html;
        mx.drawn.set(key, html);
      }
    }
    return;
  }
  table.dataset.shape = shape;
  mx.drawn = new Map();
  if (!rows.length) {
    table.innerHTML = `<tbody><tr><td class="mx-empty">No indexed papers here yet.</td></tr></tbody>`;
    return;
  }
  const body = rows.map((row) => `<tr><th class="mx-paper" scope="row"><strong>${esc(row.cite)}</strong><span>${esc(row.title)}</span>
    <button type="button" class="link-btn" data-explain-doc="${esc(row.key)}">Explain this paper</button></th>
    ${columns.map((column) => {
      const html = matrixCellHtml(row, column);
      mx.drawn.set(`${row.key}|${column.id}`, html);
      return html;
    }).join("")}</tr>`).join("");
  table.innerHTML = `<thead><tr><th class="mx-paper" scope="col">Paper</th>${columns.map((column) => `
    <th scope="col" title="${esc(column.question)}"><div class="mx-head"><span>${esc(column.label)}</span>
      <button type="button" class="icon-btn" data-mx-remove="${column.id}" title="Remove this column" aria-label="Remove the ${esc(column.label)} column"><svg width="12" height="12"><use href="#i-x"/></svg></button></div></th>`).join("")}</tr></thead>
    <tbody>${body}</tbody>`;
}

function renderMatrix() {
  const { columns, job } = mx.data;
  renderMatrixFilter();
  const rows = matrixRows();
  const pending = rows.reduce((n, row) => n + columns.filter((column) => {
    const cell = matrixCell(row.key, column.id);
    return !cell || cell.stale;
  }).length, 0);

  const fill = $("mx-fill");
  fill.disabled = job.running || !pending;
  fill.textContent = job.running ? "Filling" : pending ? `Fill ${plural(pending, "empty cell")}` : "All cells filled";
  const minutes = Math.max(1, Math.round((pending * (job.seconds_per_cell || 3.5)) / 60));
  fill.title = pending ? `About ${minutes} minute${minutes === 1 ? "" : "s"} on the model on this PC. You can keep using Folio meanwhile.` : "";

  $("mx-progress").hidden = !job.running;
  if (job.running) {
    $("mx-progress-label").textContent = `Filled ${job.done} of ${job.total}${job.current ? `. Now: ${job.current}` : ""}`;
    $("mx-bar").style.width = `${job.total ? (100 * job.done) / job.total : 0}%`;
  }
  $("mx-export").href = `/api/matrix/export?keys=${encodeURIComponent(rows.map((row) => row.key).join(","))}`;
  const dupes = mx.data.duplicates || 0;
  $("mx-dupes").hidden = !dupes;
  $("mx-dupes").textContent = dupes
    ? `${plural(dupes, "paper")} ${dupes === 1 ? "is" : "are"} in Zotero twice, so the matrix shows only the copy with the PDF. To tidy up, open Zotero, click Duplicate Items in the left pane, and merge each pair.`
    : "";

  renderMatrixTable(rows, columns);

  const labels = new Set(columns.map((column) => column.label.toLowerCase()));
  const ideas = MX_IDEAS.map((idea, index) => [idea, index]).filter(([[label]]) => !labels.has(label.toLowerCase()));
  $("mx-ideas").innerHTML = ideas.length
    ? `Column ideas for your thesis:${ideas.map(([[label], index]) => `<button type="button" class="link-btn" data-mx-idea="${index}">${esc(label)}</button>`).join("")}`
    : "";
}

async function fillMatrix() {
  const keys = matrixRows().map((row) => row.key);
  try {
    const { started } = await api("/api/matrix/fill", { body: { doc_keys: keys } });
    if (!started) toast("The matrix is already being filled.");
    mx.wasRunning = true;
    await refreshMatrix();
  } catch (error) {
    toast(error.message);
  }
}

async function addMatrixColumn(event) {
  event.preventDefault();
  const label = $("mx-label").value.trim();
  const question = $("mx-question").value.trim();
  if (!label || question.length < 5) return toast("Give the column a name and say what to look for in each paper.");
  try {
    await api("/api/matrix/columns", { body: { label, question } });
    $("mx-label").value = "";
    $("mx-question").value = "";
    await refreshMatrix();
    toast(`Added "${label}". Fill the empty cells to complete it.`);
  } catch (error) {
    toast(error.message);
  }
}

async function removeMatrixColumn(columnId) {
  const column = mx.data.columns.find((c) => c.id === columnId);
  if (!column || !confirm(`Remove the "${column.label}" column and its notes?`)) return;
  await api(`/api/matrix/columns/${columnId}`, { method: "DELETE" });
  await refreshMatrix();
}

function openMatrixSource(target) {
  const row = mx.data.rows.find((r) => r.key === target.dataset.mxDoc);
  const cell = matrixCell(target.dataset.mxDoc, Number(target.dataset.mxCol));
  const source = cell?.sources.find((s) => s.n === Number(target.dataset.mxN));
  if (!row || !source) return;
  openViewer({ ...source, doc_key: row.key, cite: row.cite, title: row.title, has_pdf: row.has_pdf, pages: row.pages }, target);
}

function removeScope(kind, value) {
  if (kind === "years") { state.scope.year_min = null; state.scope.year_max = null; }
  else state.scope[kind] = state.scope[kind].filter((entry) => entry !== value);
  renderScopeChips();
  if (!$("scope-pop").hidden) renderScopePopover();
}

function closeScope() {
  $("scope-pop").hidden = true;
  $("scope-btn").setAttribute("aria-expanded", "false");
}

$("scope-btn").addEventListener("click", async () => {
  const pop = $("scope-pop");
  if (!pop.hidden) return closeScope();
  await renderScopePopover();
  pop.hidden = false;
  $("scope-btn").setAttribute("aria-expanded", "true");
});

$("scope-pop").addEventListener("change", (event) => {
  const input = event.target;
  if (input.dataset.kind) {
    const list = state.scope[input.dataset.kind];
    state.scope[input.dataset.kind] = input.checked ? [...list, input.value] : list.filter((v) => v !== input.value);
  } else if (input.id === "scope-from") {
    state.scope.year_min = Number(input.value) || null;
  } else if (input.id === "scope-to") {
    state.scope.year_max = Number(input.value) || null;
  }
  renderScopeChips();
});

$("modes").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (button) setMode(button.dataset.mode);
});

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#modes button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) stopAnswer();
  else ask($("question").value);
});
$("question").addEventListener("input", autoGrow);
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
$("new-chat").addEventListener("click", newConversation);
$("model-select").addEventListener("change", async (event) => {
  const model = event.target.value;
  if (isClaude(model) && !state.status.claude.has_key) {
    event.target.value = state.status.llm.selected;
    openKeyDialog(model);
    return;
  }
  if (missingModel(model)) {
    event.target.value = state.status.llm.selected;
    downloadModel(model);
    return;
  }
  try {
    await chooseModel(model);
  } catch (error) {
    event.target.value = state.status.llm.selected;
    toast(error.message);
  }
});
$("manage-key").addEventListener("click", () => openKeyDialog());
$("model-get").addEventListener("click", () => downloadModel(state.status.llm.selected));
$("key-save").addEventListener("click", saveKey);
$("key-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); saveKey(); }
});
$("key-cancel").addEventListener("click", () => $("key-dialog").close());
$("key-remove").addEventListener("click", () => removeKey().catch((error) => toast(error.message)));
$("lib-filter").addEventListener("input", renderLibraryList);
$("sync-now").addEventListener("click", async () => {
  const { started } = await api("/api/sync", { method: "POST" });
  toast(started ? "Syncing with Zotero." : "A sync is already running.");
});
$("open-zotero").addEventListener("click", launchZotero);
$("disc-form").addEventListener("submit", runDiscover);
$("claim-form").addEventListener("submit", findSources);
$("mx-fill").addEventListener("click", fillMatrix);
$("mx-stop").addEventListener("click", () => api("/api/matrix/stop", { method: "POST" }).catch((error) => toast(error.message)));
$("mx-add").addEventListener("submit", addMatrixColumn);
$("mx-filter").addEventListener("change", (event) => { mx.filter = event.target.value; renderMatrix(); });
$("claim-text").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("claim-form").requestSubmit();
  }
});
$("v-explain").addEventListener("click", () => {
  if (!state.viewer) return;
  const key = state.viewer.source.doc_key;
  if (matchMedia("(max-width: 1080px)").matches) closeViewer(); // In a small window the page would cover the explanation.
  explainDoc(key).catch((error) => toast(error.message));
});
$("v-close").addEventListener("click", closeViewer);
$("v-prev").addEventListener("click", () => { state.viewer.page -= 1; showViewerPage(); });
$("v-next").addEventListener("click", () => { state.viewer.page += 1; showViewerPage(); });
$("v-zotero").addEventListener("click", () => {
  if (state.viewer) openInZotero(state.viewer.source.doc_key, state.viewer.source.has_pdf ? state.viewer.page : null);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (!$("scope-pop").hidden) closeScope();
    else if (state.viewer) closeViewer();
  }
});

// ------------------------------------------------------------------------ start

setMode("answer");
renderScopeChips();
showView("ask");
updateSend();
refreshStatus();
refreshConversations().catch(() => {});
