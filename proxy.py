import os
import json
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

NVIDIA_API_KEY = os.environ.get("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", "").strip()
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "fevrier", "mars", "avril", "mai", "juin",
        "juillet", "aout", "septembre", "octobre", "novembre", "decembre"]


def get_date_fr():
    now = datetime.now()
    return f"{JOURS[now.weekday()]} {now.day} {MOIS[now.month - 1]} {now.year}"


@app.route("/api/chat", methods=["POST"])
def chat():
    if not NVIDIA_API_KEY:
        def err_gen():
            yield 'data: {"error": "Cle API NVIDIA manquante sur le serveur."}\n\n'.encode("utf-8")
        return Response(stream_with_context(err_gen()), mimetype="text/event-stream")

    data = request.get_json()
    messages = data.get("messages", [])
    config = data.get("config", {})

    # Configuration personnalisée depuis le frontend
    ai_name = config.get("aiName", "NOVA")
    personality = config.get("personality", "Tu es un assistant IA utile et bienveillant.")
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("maxTokens", 3000)
    mode = data.get("mode", "moyen")

    # Ajustements selon le mode
    mode_config = {
        "faible": {"max": 1500, "system_suffix": " Reponds de facon concise en 1 a 3 phrases."},
        "moyen":  {"max": 3000, "system_suffix": " Sois clair et equilibre."},
        "max":    {"max": 8192, "system_suffix": " Analyse en profondeur, structure ta reponse."}
    }
    m_config = mode_config.get(mode, mode_config["moyen"])
    max_tokens = min(max_tokens, m_config["max"])

    date_str = get_date_fr()
    system_prompt = (
        f"Tu t'appelles {ai_name}. {personality}"
        f" Tu reponds TOUJOURS en francais, sans exception."
        f"{m_config['system_suffix']}"
        f" La date actuelle est le {date_str}."
    )

    full_messages = [
        {"role": "system", "content": system_prompt},
        *messages
    ]

    payload = {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "messages": full_messages,
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
    key_len = len(NVIDIA_API_KEY) if NVIDIA_API_KEY else 0
    print(f"[NOVA] Cle API chargee : {key_len} caracteres")
    app.run(host="0.0.0.0", port=port, debug=False)