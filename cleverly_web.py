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
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2000"))  # characters
HEARTBEAT_INTERVAL = 15  # seconds between keep-alive SSE comments

# --- App setup ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret")
logging.basicConfig(level=logging.INFO)


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


@app.route("/chat")
def stream():
    raw_prompt = request.args.get("msg", "")
    model_key = request.args.get("model", "normal")
    model = _validate_model(model_key)
    prompt = _sanitize_prompt(raw_prompt)

    # Build messages for Ollama
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
        import time
        last_heartbeat = time.time()

        try:
            # Stream from Ollama
            stream_iter = chat(model=model, messages=messages, stream=True)

            for chunk in stream_iter:
                last_heartbeat = time.time()
                content = ""
                try:
                    content = chunk.get("message", {}).get("content", "")
                except Exception:
                    content = str(chunk)

                if content:
                    payload = {"content": content}
                    yield f"data: {json.dumps(payload)}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as exc:
            current_app.logger.exception("Streaming error")
            err_payload = {"error": "Server error while streaming response.", "detail": str(exc)}
            yield f"data: {json.dumps(err_payload)}\n\n"

        yield ": stream closed\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
