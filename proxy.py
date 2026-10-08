import os
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# Lecture de la cle
_RAW_KEY = os.environ.get("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", "")
NVIDIA_API_KEY = _RAW_KEY.strip() if _RAW_KEY else ""
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Debug au demarrage
print("=" * 60, flush=True)
print(f"[NOVA] KEY RECUE : '{NVIDIA_API_KEY[:15]}...' (longueur={len(NVIDIA_API_KEY)})", flush=True)
print(f"[NOVA] KEY VALIDE : {NVIDIA_API_KEY.startswith('nvapi-') and len(NVIDIA_API_KEY) > 50}", flush=True)
print("=" * 60, flush=True)


@app.route("/debug")
def debug():
    """Endpoint de diagnostic. Ne montre PAS la clé complète."""
    return {
        "key_present": bool(NVIDIA_API_KEY),
        "key_length": len(NVIDIA_API_KEY),
        "key_is_valid_format": NVIDIA_API_KEY.startswith("nvapi-") and len(NVIDIA_API_KEY) > 50,
        "env_var_exists": "NVIDIA_API_KEY" in os.environ,
        "message": "OK" if NVIDIA_API_KEY else "La variable NVIDIA_API_KEY est vide ou absente"
    }


@app.route("/api/chat", methods=["POST"])
def chat():
    # VERIFICATION CLAIRE
    if not NVIDIA_API_KEY or len(NVIDIA_API_KEY) < 50:
        def err_gen():
            msg = f"Cle API invalide (longueur={len(NVIDIA_API_KEY)})"
            yield f'data: {{"error": "{msg}"}}\n\n'.encode("utf-8")
        return Response(stream_with_context(err_gen()), mimetype="text/event-stream")

    data = request.get_json()
    messages = data.get("messages", [])
    config = data.get("config", {})

    ai_name = config.get("aiName", "NOVA")
    personality = config.get("personality", "Tu es un assistant IA utile.")
    temperature = float(config.get("temperature", 0.7))
    mode = data.get("mode", "moyen")

    mode_limits = {"faible": 1000, "moyen": 2500, "max": 4000}
    max_tokens = min(int(config.get("maxTokens", 2500)), mode_limits.get(mode, 2500))

    now = datetime.now()
    date_str = f"{now.day}/{now.month}/{now.year}"
    system_prompt = (
        f"Tu t'appelles {ai_name}. {personality} "
        f"Tu reponds TOUJOURS en francais. Nous sommes le {date_str}."
    )

    payload = {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
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
            print(f"[NOVA] Envoi requete a NVIDIA...", flush=True)
            with requests.post(NVIDIA_URL, json=payload, headers=headers, stream=True, timeout=(10, 90)) as r:
                print(f"[NOVA] Reponse NVIDIA status: {r.status_code}", flush=True)
                if r.status_code != 200:
                    err = r.text[:200].replace('"', "'").replace("\n", " ")
                    yield f'data: {{"error": "NVIDIA API {r.status_code}: {err}"}}\n\n'.encode("utf-8")
                    return
                for line in r.iter_lines():
                    if line:
                        yield line + b"\n"
        except requests.exceptions.Timeout:
            yield b'data: {"error": "Timeout - NVIDIA a mis trop de temps"}\n\n'
        except Exception as e:
            err = str(e)[:200].replace('"', "'").replace("\n", " ")
            yield f'data: {{"error": "Erreur: {err}"}}\n\n'.encode("utf-8")

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/")
def index():
    return send_from_directory(".", "nova.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
