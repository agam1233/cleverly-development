# cleverly_web.py
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context
import json
import os
import threading
import time
import uuid
from ollama import chat

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret")

# Model choices must match your local Ollama models.
MODEL_CHOICES = {
    "normal": {"id": "qwen3:8b", "label": "Normal (Qwen 8B)"},
    "balanced": {"id": "llama3:8b", "label": "Balanced (Llama 3 8B)"},
    "fast": {"id": "ministral-3:3b", "label": "Fast (Ministral 3B)"},
}

DEFAULT_SYSTEM_PROMPT = (
    "You are Cleverly, a calm and helpful AI assistant. "
    "Answer clearly and concisely. Use short paragraphs and practical examples when useful."
)
MAX_PROMPT_LENGTH = 6000
MAX_SYSTEM_PROMPT_LENGTH = 4000
MAX_HISTORY_MESSAGES = 24
MAX_SESSIONS = 200

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 512

CHAT_SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


def normalize_text(value, max_len):
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").strip()
    if len(text) > max_len:
        text = text[:max_len]
    return text


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def clamp_float(value, minimum, maximum, default):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def clamp_int(value, minimum, maximum, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def sanitize_chat_id(value, create_if_empty=True):
    raw = (value or "").strip()
    if not raw:
        return str(uuid.uuid4()) if create_if_empty else ""
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in {"-", "_"})
    if not safe:
        return str(uuid.uuid4()) if create_if_empty else ""
    return safe[:64]


def trim_history(messages):
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return
    del messages[:-MAX_HISTORY_MESSAGES]
    while messages and messages[0].get("role") == "assistant":
        messages.pop(0)


def cleanup_sessions_unlocked():
    if len(CHAT_SESSIONS) <= MAX_SESSIONS:
        return
    oldest = sorted(
        CHAT_SESSIONS.items(),
        key=lambda item: item[1].get("last_seen", 0.0),
    )
    drop_count = len(CHAT_SESSIONS) - MAX_SESSIONS
    for chat_id, _ in oldest[:drop_count]:
        CHAT_SESSIONS.pop(chat_id, None)


def get_payload():
    if request.method == "POST":
        return request.get_json(silent=True) or {}
    return request.args.to_dict(flat=True)


def sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.route("/")
def index():
    return render_template_string(HTML, default_system_prompt=DEFAULT_SYSTEM_PROMPT)


@app.get("/models")
def models():
    return jsonify(
        [
            {"key": key, "model": item["id"], "label": item["label"]}
            for key, item in MODEL_CHOICES.items()
        ]
    )


@app.post("/reset")
def reset_chat():
    payload = request.get_json(silent=True) or {}
    chat_id = sanitize_chat_id(payload.get("chat_id", ""), create_if_empty=False)
    cleared = False
    if chat_id:
        with SESSIONS_LOCK:
            cleared = CHAT_SESSIONS.pop(chat_id, None) is not None
    return jsonify({"ok": True, "chat_id": chat_id, "cleared": cleared})


@app.route("/chat", methods=["GET", "POST"])
def stream():
    payload = get_payload()

    chat_id = sanitize_chat_id(payload.get("chat_id", ""))
    model_key = normalize_text(payload.get("model", "normal"), 32) or "normal"
    model_info = MODEL_CHOICES.get(model_key, MODEL_CHOICES["normal"])
    model_name = model_info["id"]

    system_prompt = normalize_text(
        payload.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        MAX_SYSTEM_PROMPT_LENGTH,
    )
    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    temperature = clamp_float(
        payload.get("temperature", DEFAULT_TEMPERATURE),
        0.0,
        2.0,
        DEFAULT_TEMPERATURE,
    )
    top_p = clamp_float(payload.get("top_p", DEFAULT_TOP_P), 0.0, 1.0, DEFAULT_TOP_P)
    max_tokens = clamp_int(
        payload.get("max_tokens", DEFAULT_MAX_TOKENS),
        64,
        2048,
        DEFAULT_MAX_TOKENS,
    )

    prompt = normalize_text(payload.get("msg", ""), MAX_PROMPT_LENGTH)
    regenerate = parse_bool(payload.get("regenerate", False))

    with SESSIONS_LOCK:
        state = CHAT_SESSIONS.setdefault(
            chat_id,
            {"messages": [], "last_user": "", "last_seen": time.time()},
        )
        state["last_seen"] = time.time()
        cleanup_sessions_unlocked()

        if regenerate:
            if state["messages"] and state["messages"][-1]["role"] == "assistant":
                state["messages"].pop()
            if not prompt:
                prompt = state.get("last_user", "")
            if prompt:
                if not state["messages"] or state["messages"][-1]["role"] != "user":
                    state["messages"].append({"role": "user", "content": prompt})
                elif state["messages"][-1]["content"] != prompt:
                    state["messages"].append({"role": "user", "content": prompt})
                state["last_user"] = prompt
        else:
            if prompt:
                state["messages"].append({"role": "user", "content": prompt})
                state["last_user"] = prompt

        trim_history(state["messages"])
        context_messages = [item.copy() for item in state["messages"]]

    if not prompt:
        def empty_stream():
            yield sse(
                {
                    "error": "Message is empty. Provide text or regenerate an existing response.",
                    "done": True,
                }
            )

        return Response(
            stream_with_context(empty_stream()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    messages = [{"role": "system", "content": system_prompt}] + context_messages

    def generate():
        started = time.time()
        assistant_parts = []
        approx_tokens = 0

        yield sse({"meta": {"chat_id": chat_id, "model": model_name}})
        try:
            stream_iter = chat(
                model=model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
            )
            for chunk in stream_iter:
                content = ""
                if isinstance(chunk, dict):
                    content = chunk.get("message", {}).get("content", "") or ""
                if not content:
                    continue
                assistant_parts.append(content)
                approx_tokens += max(1, len(content) // 4)
                yield sse({"content": content})

            assistant_text = "".join(assistant_parts).strip()
            if assistant_text:
                with SESSIONS_LOCK:
                    live = CHAT_SESSIONS.setdefault(
                        chat_id,
                        {"messages": [], "last_user": "", "last_seen": time.time()},
                    )
                    live["messages"].append({"role": "assistant", "content": assistant_text})
                    live["last_seen"] = time.time()
                    trim_history(live["messages"])

            elapsed_ms = int((time.time() - started) * 1000)
            yield sse(
                {
                    "done": True,
                    "meta": {
                        "elapsed_ms": elapsed_ms,
                        "approx_tokens": approx_tokens,
                        "chat_id": chat_id,
                    },
                }
            )
        except Exception as exc:
            yield sse({"error": str(exc), "done": True})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Cleverly - Local AI Chat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg-0:#030712;
  --bg-1:#0b1225;
  --panel:#11182c;
  --panel-2:#0b1325;
  --line:rgba(255,255,255,0.08);
  --text:#e7edf9;
  --muted:#9ba8c7;
  --accent:#4f7cff;
  --warn:#f59e0b;
  --danger:#f87171;
  --radius:14px;
}
*{box-sizing:border-box}
html,body{
  height:100%;
  margin:0;
  font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
  background:radial-gradient(circle at 20% 0%, #14213d 0%, #030712 45%, #020617 100%);
  color:var(--text);
}
.app{
  max-width:1180px;
  margin:24px auto;
  display:grid;
  grid-template-columns:320px 1fr;
  gap:18px;
  padding:16px;
}
.panel{
  background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
  border:1px solid var(--line);
  border-radius:var(--radius);
  box-shadow:0 14px 38px rgba(2,6,23,0.55);
  overflow:hidden;
}
.sidebar{
  padding:16px;
  display:flex;
  flex-direction:column;
  gap:12px;
}
.header{
  display:flex;
  justify-content:space-between;
  align-items:center;
}
.brand{
  display:flex;
  align-items:center;
  gap:10px;
}
.logo{
  width:34px;
  height:34px;
  border-radius:9px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:700;
  background:linear-gradient(135deg,#4f7cff,#2dd4bf);
  color:#fff;
}
.small{
  font-size:12px;
  color:var(--muted);
}
.kv{
  font-size:13px;
  color:#d9e4ff;
}
.settingsBtn,.ghostBtn{
  border:1px solid var(--line);
  color:var(--muted);
  background:transparent;
  border-radius:9px;
  cursor:pointer;
  padding:8px 10px;
}
.settingsBtn:hover,.ghostBtn:hover{border-color:rgba(255,255,255,0.16);color:#d6def0}
.main{
  padding:14px;
  display:flex;
  flex-direction:column;
  height:80vh;
}
.topRow{
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:10px;
}
.meta{
  font-size:12px;
  color:var(--muted);
}
#chat{
  flex:1;
  overflow:auto;
  display:flex;
  flex-direction:column;
  gap:10px;
  padding:8px;
}
.msg{
  max-width:84%;
  border-radius:12px;
  border:1px solid var(--line);
  padding:10px 12px;
  line-height:1.52;
  white-space:pre-wrap;
  word-break:break-word;
}
.msg.user{
  align-self:flex-end;
  background:linear-gradient(180deg,#16213b,#111b33);
}
.msg.assistant{
  align-self:flex-start;
  background:linear-gradient(180deg,#0e172d,#0b1325);
}
.msgHeader{
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:.06em;
  color:var(--muted);
  margin-bottom:6px;
}
.msgText{font-size:14px}
.msgActions{
  display:flex;
  justify-content:flex-end;
  margin-top:8px;
}
.smallBtn{
  border:1px solid var(--line);
  background:transparent;
  color:var(--muted);
  border-radius:8px;
  padding:4px 8px;
  font-size:12px;
  cursor:pointer;
}
.smallBtn:hover{color:#d6def0;border-color:rgba(255,255,255,0.18)}
.typing::after{
  content:"|";
  margin-left:4px;
  color:var(--accent);
  animation:blink 1s steps(2) infinite;
}
@keyframes blink{50%{opacity:0}}
.inputBar{
  margin-top:10px;
  border:1px solid var(--line);
  border-radius:12px;
  padding:10px;
  background:rgba(255,255,255,0.02);
}
.inputBar textarea{
  width:100%;
  min-height:44px;
  max-height:220px;
  resize:none;
  border:none;
  outline:none;
  color:var(--text);
  background:transparent;
  font:inherit;
  line-height:1.45;
}
.actions{
  margin-top:8px;
  display:flex;
  gap:8px;
  justify-content:flex-end;
}
.btn{
  border:none;
  border-radius:9px;
  padding:9px 12px;
  color:#fff;
  cursor:pointer;
  font-weight:600;
}
.btn.primary{background:var(--accent)}
.btn.warn{background:#d97706}
.btn.alt{
  background:transparent;
  border:1px solid var(--line);
  color:var(--muted);
}
.btn:disabled{
  opacity:.5;
  cursor:not-allowed;
}
.footerNote{
  margin-top:auto;
  font-size:12px;
  color:var(--muted);
}
.modal-backdrop{
  position:fixed;
  inset:0;
  background:rgba(2,6,23,0.72);
  display:none;
  align-items:center;
  justify-content:center;
  z-index:50;
}
.modal{
  width:520px;
  max-width:94%;
  background:linear-gradient(180deg,#0f172a,#0b1325);
  border:1px solid var(--line);
  border-radius:12px;
  padding:16px;
}
.modal h3{margin:0 0 6px 0}
.row{
  display:flex;
  flex-direction:column;
  gap:6px;
  margin-top:10px;
}
.select,.input,.textarea{
  width:100%;
  border:1px solid var(--line);
  border-radius:9px;
  background:rgba(255,255,255,0.02);
  color:var(--text);
  padding:8px;
  font:inherit;
}
.textarea{
  min-height:90px;
  resize:vertical;
}
.rangeWrap{
  display:grid;
  grid-template-columns:1fr auto;
  gap:8px;
  align-items:center;
}
.rangeWrap input{width:100%}
.danger{
  color:var(--danger);
}
@media (max-width:980px){
  .app{grid-template-columns:1fr;padding:10px}
  .main{height:76vh}
}
</style>
</head>
<body>
  <div class="app">
    <div class="panel sidebar">
      <div class="header">
        <div class="brand">
          <div class="logo">C</div>
          <div>
            <h2 style="margin:0;font-size:18px">Cleverly</h2>
            <div class="small">Local AI with streaming</div>
          </div>
        </div>
        <button id="openSettings" class="settingsBtn" title="Settings">Settings</button>
      </div>

      <div class="small">Active model</div>
      <div id="activeModel" class="kv">Normal (Qwen 8B)</div>

      <div class="small" style="margin-top:2px">Conversation</div>
      <button id="newChatBtn" class="ghostBtn">New chat</button>

      <div class="small" style="margin-top:2px">Modern features enabled</div>
      <div class="kv">Memory, regenerate, stop, and advanced generation controls.</div>

      <div class="footerNote">Tips: Enter to send, Shift+Enter for a new line.</div>
    </div>

    <div class="panel main">
      <div class="topRow">
        <div class="meta">Streaming output</div>
        <div class="meta"><span id="status">idle</span> | <span id="stats">--</span></div>
      </div>
      <div id="chat" role="log" aria-live="polite"></div>
      <div class="inputBar">
        <textarea id="input" placeholder="Message Cleverly"></textarea>
        <div class="actions">
          <button id="stopBtn" class="btn warn" disabled>Stop</button>
          <button id="regenBtn" class="btn alt">Regenerate</button>
          <button id="sendBtn" class="btn primary">Send</button>
        </div>
      </div>
    </div>
  </div>

  <div id="modalBackdrop" class="modal-backdrop" role="dialog" aria-modal="true">
    <div class="modal" role="document" aria-labelledby="settingsTitle">
      <h3 id="settingsTitle">Settings</h3>
      <div class="small">Persisted in local storage for this browser.</div>

      <div class="row">
        <label class="small" for="modelSelect">Model</label>
        <select id="modelSelect" class="select">
          <option value="normal">Normal (Qwen 8B)</option>
          <option value="balanced">Balanced (Llama 3 8B)</option>
          <option value="fast">Fast (Ministral 3B)</option>
        </select>
      </div>

      <div class="row">
        <label class="small" for="systemPrompt">System prompt</label>
        <textarea id="systemPrompt" class="textarea">{{ default_system_prompt | e }}</textarea>
      </div>

      <div class="row">
        <label class="small" for="temperatureRange">Temperature</label>
        <div class="rangeWrap">
          <input id="temperatureRange" type="range" min="0" max="2" step="0.1" />
          <span id="temperatureValue" class="kv">0.7</span>
        </div>
      </div>

      <div class="row">
        <label class="small" for="topPRange">Top P</label>
        <div class="rangeWrap">
          <input id="topPRange" type="range" min="0" max="1" step="0.05" />
          <span id="topPValue" class="kv">0.9</span>
        </div>
      </div>

      <div class="row">
        <label class="small" for="maxTokensInput">Max tokens</label>
        <input id="maxTokensInput" class="input" type="number" min="64" max="2048" step="32" />
      </div>

      <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:14px">
        <button id="cancelSettings" class="settingsBtn">Cancel</button>
        <button id="saveSettings" class="btn primary">Save</button>
      </div>
    </div>
  </div>

<script>
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const regenBtn = document.getElementById("regenBtn");
const newChatBtn = document.getElementById("newChatBtn");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const activeModel = document.getElementById("activeModel");

const openSettings = document.getElementById("openSettings");
const modalBackdrop = document.getElementById("modalBackdrop");
const modelSelect = document.getElementById("modelSelect");
const systemPrompt = document.getElementById("systemPrompt");
const temperatureRange = document.getElementById("temperatureRange");
const topPRange = document.getElementById("topPRange");
const maxTokensInput = document.getElementById("maxTokensInput");
const temperatureValue = document.getElementById("temperatureValue");
const topPValue = document.getElementById("topPValue");
const saveSettings = document.getElementById("saveSettings");
const cancelSettings = document.getElementById("cancelSettings");

const STORAGE_SETTINGS = "cleverly_settings_v2";
const STORAGE_CHAT = "cleverly_chat_id_v2";

const DEFAULTS = {
  model: "normal",
  systemPrompt: systemPrompt.value.trim(),
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 512
};

function clampNumber(value, min, max, fallback){
  const num = Number(value);
  if(Number.isNaN(num)) return fallback;
  return Math.max(min, Math.min(max, num));
}

function makeChatId(){
  if(window.crypto && typeof window.crypto.randomUUID === "function"){
    return window.crypto.randomUUID();
  }
  return "chat_" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function readSettings(){
  let parsed = {};
  try {
    parsed = JSON.parse(localStorage.getItem(STORAGE_SETTINGS) || "{}");
  } catch (_) {}
  return {
    model: parsed.model || DEFAULTS.model,
    systemPrompt: (parsed.systemPrompt || DEFAULTS.systemPrompt).toString().trim() || DEFAULTS.systemPrompt,
    temperature: clampNumber(parsed.temperature, 0, 2, DEFAULTS.temperature),
    topP: clampNumber(parsed.topP, 0, 1, DEFAULTS.topP),
    maxTokens: Math.round(clampNumber(parsed.maxTokens, 64, 2048, DEFAULTS.maxTokens))
  };
}

function persistSettings(){
  localStorage.setItem(STORAGE_SETTINGS, JSON.stringify(settings));
}

let settings = readSettings();
let chatId = localStorage.getItem(STORAGE_CHAT) || makeChatId();
localStorage.setItem(STORAGE_CHAT, chatId);

let currentBotEl = null;
let buffer = "";
let rafPending = false;
let abortController = null;
let isStreaming = false;

function nearBottom(){
  return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 120;
}

function scrollBottom(force = false){
  if(force || nearBottom()){
    chat.scrollTop = chat.scrollHeight;
  }
}

function modelLabelFromKey(key){
  const opt = modelSelect.querySelector(`option[value="${key}"]`);
  return opt ? opt.textContent : "Normal (Qwen 8B)";
}

function refreshModelLabel(){
  activeModel.textContent = modelLabelFromKey(settings.model);
}

function setStatus(text){
  statusEl.textContent = text;
}

function updateButtons(){
  sendBtn.disabled = isStreaming;
  stopBtn.disabled = !isStreaming;
  regenBtn.disabled = isStreaming;
}

function appendMessage(role, text = ""){
  const shouldStick = nearBottom();
  const el = document.createElement("article");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="msgHeader">${role === "user" ? "You" : "Cleverly"}</div><div class="msgText"></div>`;
  const textEl = el.querySelector(".msgText");
  textEl.textContent = text;

  if(role === "assistant"){
    const actions = document.createElement("div");
    actions.className = "msgActions";
    const copyBtn = document.createElement("button");
    copyBtn.className = "smallBtn";
    copyBtn.type = "button";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", async () => {
      const original = copyBtn.textContent;
      try {
        await navigator.clipboard.writeText(textEl.textContent || "");
        copyBtn.textContent = "Copied";
      } catch (_) {
        copyBtn.textContent = "Failed";
      } finally {
        setTimeout(() => { copyBtn.textContent = original; }, 900);
      }
    });
    actions.appendChild(copyBtn);
    el.appendChild(actions);
  }

  chat.appendChild(el);
  scrollBottom(shouldStick);
  return el;
}

function flushBuffer(){
  if(currentBotEl && buffer){
    const textEl = currentBotEl.querySelector(".msgText");
    textEl.textContent += buffer;
    buffer = "";
    scrollBottom();
  }
  rafPending = false;
}

function queueToken(text){
  buffer += text;
  if(!rafPending){
    rafPending = true;
    requestAnimationFrame(flushBuffer);
  }
}

function processEventBlock(block){
  const lines = block.split("\n").filter((line) => line.startsWith("data:"));
  if(!lines.length) return;

  const payload = lines.map((line) => line.slice(5).trim()).join("\n");
  let data;
  try {
    data = JSON.parse(payload);
  } catch (_) {
    return;
  }

  if(data.content){
    queueToken(data.content);
  }

  if(data.error){
    flushBuffer();
    if(currentBotEl){
      const textEl = currentBotEl.querySelector(".msgText");
      textEl.textContent += `\n\n[Error] ${data.error}`;
      currentBotEl.classList.remove("typing");
    }
    setStatus("error");
  }

  if(data.meta){
    if(data.meta.chat_id && data.meta.chat_id !== chatId){
      chatId = data.meta.chat_id;
      localStorage.setItem(STORAGE_CHAT, chatId);
    }
    const elapsed = Number(data.meta.elapsed_ms);
    const approx = Number(data.meta.approx_tokens);
    const parts = [];
    if(!Number.isNaN(elapsed) && elapsed > 0) parts.push(`${elapsed} ms`);
    if(!Number.isNaN(approx) && approx > 0) parts.push(`~${approx} tok`);
    if(parts.length) statsEl.textContent = parts.join("  ");
  }

  if(data.done){
    flushBuffer();
    if(currentBotEl) currentBotEl.classList.remove("typing");
    if(statusEl.textContent !== "error"){
      setStatus("idle");
    }
  }
}

function parseSSE(bufferText){
  let work = bufferText.replace(/\r/g, "");
  let marker = work.indexOf("\n\n");
  while(marker !== -1){
    const block = work.slice(0, marker).trim();
    work = work.slice(marker + 2);
    if(block) processEventBlock(block);
    marker = work.indexOf("\n\n");
  }
  return work;
}

async function streamChat(payload){
  if(abortController){
    abortController.abort();
    abortController = null;
  }

  abortController = new AbortController();
  isStreaming = true;
  updateButtons();
  setStatus("streaming...");

  const body = {
    chat_id: chatId,
    model: settings.model,
    system_prompt: settings.systemPrompt,
    temperature: settings.temperature,
    top_p: settings.topP,
    max_tokens: settings.maxTokens,
    regenerate: Boolean(payload.regenerate),
    msg: payload.msg || ""
  };

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
      signal: abortController.signal
    });
    if(!resp.ok || !resp.body){
      throw new Error(`Request failed (${resp.status})`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";

    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      pending += decoder.decode(value, {stream: true});
      pending = parseSSE(pending);
    }
    if(pending.trim()){
      parseSSE(pending + "\n\n");
    }
  } catch (err) {
    if(err && err.name === "AbortError"){
      setStatus("stopped");
    } else {
      const message = err && err.message ? err.message : "Unknown error";
      setStatus("error");
      if(currentBotEl){
        const textEl = currentBotEl.querySelector(".msgText");
        textEl.textContent += `\n\n[Error] ${message}`;
        currentBotEl.classList.remove("typing");
      }
    }
  } finally {
    flushBuffer();
    if(currentBotEl) currentBotEl.classList.remove("typing");
    isStreaming = false;
    abortController = null;
    updateButtons();
    if(statusEl.textContent === "streaming..."){
      setStatus("idle");
    }
  }
}

function send(){
  if(isStreaming) return;
  const text = input.value.trim();
  if(!text) return;

  input.value = "";
  autoResizeInput();

  appendMessage("user", text);
  currentBotEl = appendMessage("assistant", "");
  currentBotEl.classList.add("typing");
  statsEl.textContent = "--";
  setStatus("connecting...");

  streamChat({msg: text, regenerate: false});
}

function stop(){
  if(abortController){
    abortController.abort();
  }
}

function regenerate(){
  if(isStreaming) return;
  if(!chat.querySelector(".msg.user")) return;

  const assistants = chat.querySelectorAll(".msg.assistant");
  if(assistants.length){
    assistants[assistants.length - 1].remove();
  }

  currentBotEl = appendMessage("assistant", "");
  currentBotEl.classList.add("typing");
  statsEl.textContent = "--";
  setStatus("connecting...");

  streamChat({msg: "", regenerate: true});
}

async function newChat(){
  stop();
  const previousChatId = chatId;
  chatId = makeChatId();
  localStorage.setItem(STORAGE_CHAT, chatId);
  chat.innerHTML = "";
  currentBotEl = null;
  statsEl.textContent = "--";
  setStatus("idle");

  try {
    await fetch("/reset", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({chat_id: previousChatId})
    });
  } catch (_) {}
}

function openModal(){
  modelSelect.value = settings.model;
  systemPrompt.value = settings.systemPrompt;
  temperatureRange.value = String(settings.temperature);
  topPRange.value = String(settings.topP);
  maxTokensInput.value = String(settings.maxTokens);
  temperatureValue.textContent = Number(settings.temperature).toFixed(1);
  topPValue.textContent = Number(settings.topP).toFixed(2);
  modalBackdrop.style.display = "flex";
}

function closeModal(){
  modalBackdrop.style.display = "none";
}

function saveSettingsFromModal(){
  settings = {
    model: modelSelect.value || DEFAULTS.model,
    systemPrompt: systemPrompt.value.trim() || DEFAULTS.systemPrompt,
    temperature: clampNumber(temperatureRange.value, 0, 2, DEFAULTS.temperature),
    topP: clampNumber(topPRange.value, 0, 1, DEFAULTS.topP),
    maxTokens: Math.round(clampNumber(maxTokensInput.value, 64, 2048, DEFAULTS.maxTokens))
  };
  persistSettings();
  refreshModelLabel();
  closeModal();
}

function autoResizeInput(){
  input.style.height = "auto";
  input.style.height = Math.min(220, input.scrollHeight) + "px";
}

sendBtn.addEventListener("click", send);
stopBtn.addEventListener("click", stop);
regenBtn.addEventListener("click", regenerate);
newChatBtn.addEventListener("click", newChat);

input.addEventListener("input", autoResizeInput);
input.addEventListener("keydown", (e) => {
  if(e.key === "Enter" && !e.shiftKey){
    e.preventDefault();
    send();
  }
});

openSettings.addEventListener("click", openModal);
cancelSettings.addEventListener("click", closeModal);
saveSettings.addEventListener("click", saveSettingsFromModal);
modalBackdrop.addEventListener("click", (e) => {
  if(e.target === modalBackdrop){
    closeModal();
  }
});

temperatureRange.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperatureRange.value).toFixed(1);
});

topPRange.addEventListener("input", () => {
  topPValue.textContent = Number(topPRange.value).toFixed(2);
});

refreshModelLabel();
updateButtons();
setStatus("idle");
autoResizeInput();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))


