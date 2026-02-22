# app.py
from flask import Flask, request, Response, render_template_string, stream_with_context, current_app
import json
import os
import html
import logging
from ollama import chat

# --- Configuration ---
MODEL_CHOICES = {
    "normal": os.getenv("MODEL_NORMAL", "qwen3:8b"),
    "balanced": os.getenv("MODEL_BALANCED", "llama3:8b"),
    "fast": os.getenv("MODEL_FAST", "codellama:7b-instruct"),
}
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2000"))  # characters
HEARTBEAT_INTERVAL = 15  # seconds between keep-alive SSE comments

# --- App setup ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret")
logging.basicConfig(level=logging.INFO)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


def _validate_model(key: str) -> str:
    """Return a safe model string from MODEL_CHOICES; default to normal."""
    return MODEL_CHOICES.get(key, MODEL_CHOICES["normal"])


def _sanitize_prompt(raw: str) -> str:
    """Trim, enforce length limit, and escape HTML to avoid XSS."""
    if raw is None:
        return ""
    prompt = raw.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        prompt = prompt[:MAX_PROMPT_LENGTH]
    return html.escape(prompt)


@app. route("/chat")
def stream():
    raw_prompt = request.args.get("msg", "")
    model_key = request.args.get("model", "normal")
    model = _validate_model(model_key)
    prompt = _sanitize_prompt(raw_prompt)

    # Build messages with a clear system instruction
    messages = [
        {
            "role": "system",
            "content": (
                "You are Cleverly, a chill and helpful AI assistant. "
                "Give concise, smart answers formatted in clear paragraphs. "
                "Avoid asterisks, hashtags, or masking characters."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    def generate():
        # SSE heartbeat comment to keep proxies alive
        import time
        last_heartbeat = time.time()

        try:
            # Start streaming from Ollama
            stream_iter = chat(model=model, messages=messages, stream=True)

            # If the stream yields incremental chunks, append them to the client.
            for chunk in stream_iter:
                # Reset heartbeat timer on each chunk
                last_heartbeat = time.time()

                # Defensive extraction: chunk shape may vary; handle missing keys
                content = ""
                try:
                    # expected shape: {'message': {'content': '...'}}
                    content = chunk.get("message", {}).get("content", "")
                except Exception:
                    content = str(chunk)

                if content:
                    # Send SSE data event with JSON payload
                    payload = {"content": content}
                    yield f"data: {json.dumps(payload)}\n\n"

                # Optionally yield heartbeat if no data for a while (handled above)
                # continue streaming

            # After stream completes, send a final event indicating done
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as exc:
            current_app.logger.exception("Streaming error")
            err_payload = {"error": "Server error while streaming response."}
            # Include a short error message for debugging (escaped)
            err_payload["detail"] = str(exc)
            yield f"data: {json.dumps(err_payload)}\n\n"

        # Ensure client knows stream ended (some clients rely on this)
        # A final comment (SSE comment) and then close
        yield ": stream closed\n\n"

    # Return SSE response
    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# --- Minimal HTML client with safer streaming handling ---
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Cleverly | Ultra</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
:root{--bg:radial-gradient(circle at top,#0f172a 0%,#020617 100%);--glass:rgba(15,23,42,0.6);--border:rgba(255,255,255,0.08);--accent:#6366f1}
body{margin:0;padding:0;background:var(--bg);color:#e2e8f0;font-family:Inter,system-ui,sans-serif;min-height:100vh;display:flex;overflow:hidden}
.sidebar{width:280px;background:rgba(0,0,0,0.28);border-right:1px solid var(--border);padding:20px;backdrop-filter:blur(8px);display:flex;flex-direction:column;gap:14px}
.model-config{display:flex;flex-direction:column;gap:10px}
.main{flex:1;display:flex;flex-direction:column;padding:32px}
#chat-container{flex:1;overflow-y:auto;max-width:850px;margin:0 auto;width:100%;padding-right:12px}
.msg{margin-bottom:22px;line-height:1.7;white-space:pre-wrap;animation:fadeIn .35s;font-size:15px}
.user{color:var(--accent);font-weight:700}
.bot{color:#f1f5f9}
.typing::after{content:'▊';animation:blink 1s infinite;margin-left:6px;color:var(--accent)}
.input-box{max-width:850px;margin:18px auto 0;width:100%;background:var(--glass);border:1px solid var(--border);border-radius:12px;padding:12px;display:flex;gap:10px;box-shadow:0 12px 30px rgba(0,0,0,0.5)}
input{flex:1;background:transparent;border:none;color:white;outline:none;font-size:15px}
select{background:#1e293b;color:white;border:1px solid var(--border);border-radius:8px;padding:6px;cursor:pointer}
button{background:var(--accent);border:none;color:white;padding:8px 18px;border-radius:10px;cursor:pointer;font-weight:600}
button[disabled]{opacity:.6;cursor:not-allowed}
.legal-box{margin-top:auto;padding:12px;border:1px solid rgba(148,163,184,0.35);border-radius:10px;background:rgba(15,23,42,0.52);font-size:12px;line-height:1.45;color:#cbd5e1}
.legal-box p{margin:0 0 8px 0}
.tos-link{display:inline-block;color:#bfdbfe;text-decoration:none;font-weight:600}
.tos-link:hover{text-decoration:underline}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@keyframes blink{50%{opacity:0}}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-thumb{background:var(--border);border-radius:10px}
@media (max-width:900px){body{flex-direction:column;overflow:auto}.sidebar{width:auto;border-right:none;border-bottom:1px solid var(--border)}.main{padding:20px}.input-box{margin-top:12px}}
</style>
</head>
<body>
  <div class="sidebar">
    <div class="model-config">
      <h2 style="margin:0;letter-spacing:-1px">Cleverly</h2>
      <p style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:800;margin:0">Model Config</p>
      <select id="modelSelect" aria-label="Model selection">
        <option value="normal">Normal (Qwen)</option>
        <option value="balanced">Balanced (Llama)</option>
        <option value="fast">Fast (fastest)</option>
      </select>
    </div>
    <div class="legal-box" role="note" aria-label="Legal notice">
      <p>Cleverly is not responsible for damage.</p>
      <a class="tos-link" href="https://github.com/agam1233/cleverly-development/blob/main/README.md" target="_blank" rel="noopener noreferrer">Terms of Service</a>
    </div>
  </div>

  <div class="main">
    <div id="chat-container" role="log" aria-live="polite"></div>

    <div class="input-box">
      <input id="userInput" placeholder="Message Cleverly..." autocomplete="off" />
      <button id="sendBtn">Send</button>
    </div>
  </div>

<script>
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('userInput');
const modelSelect = document.getElementById('modelSelect');
const sendBtn = document.getElementById('sendBtn');

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, function(m){ return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]); });
}

function appendMsg(role, contentHtml = "") {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const label = role === 'user' ? 'YOU' : 'CLEVERLY';
  div.innerHTML = `<strong>${label}:</strong><div class="content" style="margin-top:6px;">${contentHtml}</div>`;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return div.querySelector('.content');
}

let currentSource = null;

function setStreamingState(isStreaming) {
  sendBtn.disabled = isStreaming;
  userInput.disabled = isStreaming;
  if (!isStreaming) userInput.focus();
}

async function send() {
  const text = userInput.value.trim();
  if (!text) return;
  userInput.value = '';
  setStreamingState(true);

  appendMsg('user', escapeHtml(text));
  const botContentEl = appendMsg('bot', '');
  botContentEl.parentElement.classList.add('typing');

  const model = modelSelect.value;
  const url = `/chat?msg=${encodeURIComponent(text)}&model=${encodeURIComponent(model)}`;

  // Close any existing source
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }

  const source = new EventSource(url);
  currentSource = source;

  source.onmessage = function(e) {
    try {
      const data = JSON.parse(e.data);
      if (data.content) {
        // Append content safely (escaped)
        botContentEl.innerHTML += escapeHtml(data.content);
        chatContainer.scrollTop = chatContainer.scrollHeight;
      } else if (data.done) {
        botContentEl.parentElement.classList.remove('typing');
        setStreamingState(false);
        source.close();
        currentSource = null;
      } else if (data.error) {
        botContentEl.innerHTML += '<div style="color:#fca5a5;margin-top:6px;">' + escapeHtml(data.error) + '</div>';
        botContentEl.parentElement.classList.remove('typing');
        setStreamingState(false);
        source.close();
        currentSource = null;
      }
    } catch (err) {
      console.error('Invalid SSE payload', err);
    }
  };

  source.onerror = function() {
    botContentEl.parentElement.classList.remove('typing');
    setStreamingState(false);
    if (source) {
      source.close();
      currentSource = null;
    }
  };
}

sendBtn.addEventListener('click', send);
userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
