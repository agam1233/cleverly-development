from flask import Flask, request, Response, stream_with_context, current_app
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
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2000"))

# --- App setup ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret")
logging.basicConfig(level=logging.INFO)

# --- Routes ---
@app.route("/")
def index():
    # Serve the HTML
    with open("cleverly.html", "r", encoding="utf-8") as f:
        return f.read()

def _validate_model(key: str) -> str:
    return MODEL_CHOICES.get(key, MODEL_CHOICES["normal"])

def _sanitize_prompt(raw: str) -> str:
    if raw is None:
        return ""
    prompt = raw.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        prompt = prompt[:MAX_PROMPT_LENGTH]
    return html.escape(prompt)

@app.route("/chat")
def stream():
    raw_prompt = request.args.get("msg", "")
    model_key = request.args.get("model", "normal")
    model = _validate_model(model_key)
    prompt = _sanitize_prompt(raw_prompt)

    messages = [
        {"role": "system", "content": "You are Cleverly, a chill and helpful AI assistant. Give concise, smart answers formatted in clear paragraphs. Avoid asterisks, hashtags, or masking characters."},
        {"role": "user", "content": prompt},
    ]

    def generate():
        import time
        try:
            stream_iter = chat(model=model, messages=messages, stream=True)
            for chunk in stream_iter:
                content = ""
                try:
                    content = chunk.get("message", {}).get("content", "")
                except Exception:
                    content = str(chunk)
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            current_app.logger.exception("Streaming error")
            yield f"data: {json.dumps({'error': 'Server error', 'detail': str(exc)})}\n\n"
        yield ": stream closed\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
