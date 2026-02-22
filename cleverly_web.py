from flask import Flask, request, Response, stream_with_context, current_app
import json, html, logging, requests, os

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret")
logging.basicConfig(level=logging.INFO)

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
MAX_PROMPT_LENGTH = 2000

# --- Model choices ---
MODEL_CHOICES = {
    "normal": "qwen3:8b",
    "balanced": "llama3:8b",
    "fast": "codellama:7b-instruct"
}

def _sanitize_prompt(raw):
    if raw is None: return ""
    return html.escape(raw.strip()[:MAX_PROMPT_LENGTH])

@app.route("/chat")
def chat_endpoint():
    raw_prompt = request.args.get("msg", "")
    model_key = request.args.get("model", "normal")
    prompt = _sanitize_prompt(raw_prompt)

    model = MODEL_CHOICES.get(model_key, MODEL_CHOICES["normal"])

    messages = [
        {"role": "system", "content": "You are Cleverly, a chill and helpful AI assistant. Give concise, smart answers."},
        {"role": "user", "content": prompt}
    ]

    def generate():
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/chat",
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
                json={"model": model, "messages": messages, "stream": True},
                stream=True,
                timeout=60
            )

            for line in response.iter_lines():
                if line:
                    yield f"data: {json.dumps({'content': line.decode()})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            current_app.logger.exception("Streaming error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield ": stream closed\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)
