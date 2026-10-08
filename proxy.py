import os
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")

# Valeurs adaptees au free tier (0.1 CPU / 512 MB / timeout ~100s)
# Surchargeables via variables d'environnement sans redeployer le code.
MODES = {
    "faible": {
        "max_tokens": int(os.environ.get("MAX_TOKENS_FAIBLE", 1500)),
        "temperature": 0.3,
        "suffix": " Reponds de facon concise."
    },
    "moyen": {
        "max_tokens": int(os.environ.get("MAX_TOKENS_MOYEN", 3000)),
        "temperature": 0.7,
        "suffix": " Sois clair et equilibre."
    },
    "max": {
        "max_tokens": int(os.environ.get("MAX_TOKENS_MAX", 5000)),
        "temperature": 1.0,
        "suffix": " Analyse en profondeur."
    }
}

# Timeout (connexion, lecture) : 90s pour rester sous le timeout plateforme
CONNECT_TIMEOUT = int(os.environ.get("CONNECT_TIMEOUT", 15))
READ_TIMEOUT = int(os.environ.get("READ_TIMEOUT", 90))


@app.route("/api/chat", methods=["POST"])
def chat():
    if not NVIDIA_API_KEY:
        def err():
            yield b'data: {"error": "Cle API manquante."}\n\n'
        return Response(stream_with_context(err()), mimetype="text/event-stream")

    data = request.get_json() or {}
    messages = data.get("messages", [])
    mode = data.get("mode", "moyen")
    config = data.get("config", {})

    cfg = MODES.get(mode, MODES["moyen"])
    ai_name = config.get("aiName", "NOVA")
    personality = config.get("personality", "Tu es un assistant IA utile et bienveillant.")
    temperature = float(config.get("temperature", cfg["temperature"]))
    max_tokens = min(int(config.get("maxTokens", cfg["max_tokens"])), cfg["max_tokens"])

    now = datetime.now()
    date_str = f"{now.day}/{now.month}/{now.year}"

    system_prompt = (
        f"Tu t'appelles {ai_name}. {personality} "
        f"Tu reponds TOUJOURS en francais. "
        f"{cfg['suffix']} Nous sommes le {date_str}."
    )

    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    def generate():
        try:
            with requests.post(
                NVIDIA_URL,
                json=payload,
                headers=headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            ) as r:
                if r.status_code != 200:
                    err = r.text[:200].replace('"', "'").replace("\n", " ")
                    yield f'data: {{"error": "API {r.status_code}: {err}"}}\n\n'.encode()
                    return
                for line in r.iter_lines():
                    if line:
                        yield line + b"\n"
        except requests.exceptions.ReadTimeout:
            yield b'data: {"error": "Delai depasse cote serveur."}\n\n'
        except Exception as e:
            err = str(e)[:200].replace('"', "'").replace("\n", " ")
            yield f'data: {{"error": "{err}"}}\n\n'.encode()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.route("/debug")
def debug():
    return {
        "key_present": bool(NVIDIA_API_KEY),
        "key_length": len(NVIDIA_API_KEY),
        "key_valid": NVIDIA_API_KEY.startswith("nvapi-") and len(NVIDIA_API_KEY) > 50,
        "model": MODEL,
        "modes": {k: v["max_tokens"] for k, v in MODES.items()},
        "timeout": {"connect": CONNECT_TIMEOUT, "read": READ_TIMEOUT}
    }


@app.route("/")
def index():
    return send_from_directory(".", "nova.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[NOVA] Cle API : {len(NVIDIA_API_KEY)} chars", flush=True)
    print(f"[NOVA] Model   : {MODEL}", flush=True)
    print(f"[NOVA] Modes   : {[(k, v['max_tokens']) for k, v in MODES.items()]}", flush=True)
    # En prod, utiliser gunicorn (voir ci-dessous). app.run = dev uniquement.
    app.run(host="0.0.0.0", port=port, debug=False)
