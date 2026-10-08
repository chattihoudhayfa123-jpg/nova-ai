import os
import json
import time
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

NVIDIA_API_KEY = os.environ.get("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", "").strip()
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Liste de modèles de secours (essayés dans l'ordre si le premier échoue)
MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-nemo-12b-instruct",
]

print("=" * 60, flush=True)
print(f"[NOVA BOOT] Cle API : {'OK' if NVIDIA_API_KEY else 'MANQUANTE'} ({len(NVIDIA_API_KEY)} chars)", flush=True)
print("=" * 60, flush=True)


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
        "faible": {"max": 1000, "suffix": " Reponds de facon concise."},
        "moyen":  {"max": 2500, "suffix": " Sois clair et equilibre."},
        "max":    {"max": 4000, "suffix": " Analyse en profondeur."}
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
        # Send a keepalive immediately so the browser knows connection is alive
        yield b": keepalive\n\n"

        # Try each model in order
        last_error = None
        for model in MODELS:
            try:
                payload["model"] = model
                print(f"[NOVA] Tentative avec modele: {model}", flush=True)

                with requests.post(
                    NVIDIA_URL,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=(10, 90)  # (connection timeout, read timeout)
                ) as r:
                    if r.status_code != 200:
                        err = r.text[:300].replace('"', "'").replace("\n", " ")
                        print(f"[NOVA] Echec {model} : {r.status_code} - {err}", flush=True)
                        last_error = f"API {r.status_code}: {err}"
                        continue  # Try next model

                    # Success - stream the response
                    print(f"[NOVA] Succes avec: {model}", flush=True)
                    got_data = False
                    for line in r.iter_lines():
                        if line:
                            got_data = True
                            yield line + b"\n"

                    if got_data:
                        return  # Done successfully

                    print(f"[NOVA] {model} : aucun contenu recu", flush=True)
                    last_error = "Aucune reponse du modele"

            except requests.exceptions.Timeout:
                print(f"[NOVA] Timeout avec {model}", flush=True)
                last_error = f"Timeout sur {model}"
            except requests.exceptions.ConnectionError as e:
                print(f"[NOVA] Erreur connexion avec {model}: {e}", flush=True)
                last_error = f"Erreur connexion"
            except Exception as e:
                print(f"[NOVA] Erreur {model}: {e}", flush=True)
                last_error = str(e)

        # All models failed
        err_msg = (last_error or "Tous les modeles ont echoue").replace('"', "'")
        yield f'data: {{"error": "Service indisponible. {err_msg}"}}\n\n'.encode("utf-8")

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.route("/")
def index():
    return send_from_directory(".", "nova.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
