import os
import json
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

# ============ DIAGNOSTIC AU DEMARRAGE ============
print("=" * 60, flush=True)
print("[NOVA BOOT] Verification de la cle API NVIDIA...", flush=True)
print(f"[NOVA BOOT] NVIDIA_API_KEY present dans env : {'NVIDIA_API_KEY' in os.environ}", flush=True)
_val = os.environ.get("NVIDIA_API_KEY", "")
print(f"[NOVA BOOT] Longueur de la cle : {len(_val)}", flush=True)
print(f"[NOVA BOOT] Commence par nvapi- : {_val.startswith('nvapi-')}", flush=True)
print("=" * 60, flush=True)
# =================================================

app = Flask(__name__)
CORS(app)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


@app.route("/debug")
def debug():
    val = os.environ.get("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", "")
    return {
        "NVIDIA_API_KEY_in_env": "NVIDIA_API_KEY" in os.environ,
        "length": len(val),
        "starts_with_nvapi": val.startswith("nvapi-"),
        "first_10_chars": val[:10] if val else "",
        "last_5_chars": val[-5:] if val else ""
    }


@app.route("/api/chat", methods=["POST"])
def chat():
    if not NVIDIA_API_KEY:
        def err_gen():
            yield 'data: {"error": "Cle API NVIDIA manquante sur le serveur."}\n\n'.encode("utf-8")
        return Response(stream_with_context(err_gen()), mimetype="text/event-stream")

    data = request.get_json()
    messages = data.get("messages", [])
    config = data.get("config", {})
    ai_name = config.get("aiName", "NOVA")
    personality = config.get("personality", "Tu es un assistant IA utile.")
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("maxTokens", 3000)
    mode = data.get("mode", "moyen")

    mode_config = {
        "faible": {"max": 1500, "suffix": " Reponds de facon concise."},
        "moyen":  {"max": 3000, "suffix": " Sois clair et equilibre."},
        "max":    {"max": 8192, "suffix": " Analyse en profondeur."}
    }
    m_config = mode_config.get(mode, mode_config["moyen"])
    max_tokens = min(max_tokens, m_config["max"])

    now = datetime.now()
    date_str = f"{now.day}/{now.month}/{now.year}"
    system_prompt = (
        f"Tu t'appelles {ai_name}. {personality}"
        f" Tu reponds TOUJOURS en francais."
        f"{m_config['suffix']}"
        f" Nous sommes le {date_str}."
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
            with requests.post(NVIDIA_URL, json=payload, headers=headers, stream=True, timeout=300) as r:
                if r.status_code != 200:
                    err = r.text[:300].replace('"', "'").replace("\n", " ")
                    yield f'data: {{"error": "API {r.status_code}: {err}"}}\n\n'.encode("utf-8")
                    return
                for line in r.iter_lines():
                    if line:
                        yield line + b"\n"
        except Exception as e:
            err = str(e).replace('"', "'").replace("\n", " ")
            yield f'data: {{"error": "Erreur serveur : {err}"}}\n\n'.encode("utf-8")

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/")
def index():
    return send_from_directory(".", "nova.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
