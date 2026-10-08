import os
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Modèles à essayer dans l'ordre
MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-nemo-12b-instruct",
]

# Modes disponibles
MODES = {
    "faible": {
        "max_tokens": 1000,
        "temperature": 0.3,
        "suffix": " Reponds de facon concise en 1 a 3 phrases maximum."
    },
    "moyen": {
        "max_tokens": 2500,
        "temperature": 0.7,
        "suffix": " Sois clair, utile et equilibre."
    },
    "max": {
        "max_tokens": 4000,
        "temperature": 1.0,
        "suffix": " Analyse en profondeur, structure ta reponse avec des titres et des listes."
    }
}

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "fevrier", "mars", "avril", "mai", "juin",
        "juillet", "aout", "septembre", "octobre", "novembre", "decembre"]


def get_key():
    """Récupère la clé API depuis l'environnement."""
    return os.environ.get("NVIDIA_API_KEY", "").strip()


# ============================================================
# DEBUG (à supprimer après)
# ============================================================
@app.route("/debug")
def debug():
    key = get_key()
    return {
        "key_present": bool(key),
        "key_length": len(key),
        "key_valid": key.startswith("nvapi-") and len(key) > 50,
        "status": "OK" if key.startswith("nvapi-") and len(key) > 50 else "La variable NVIDIA_API_KEY est vide ou invalide"
    }


# ============================================================
# CHAT
# ============================================================
@app.route("/api/chat", methods=["POST"])
def chat():
    key = get_key()

    if not key or len(key) < 50:
        def err_gen():
            yield b'data: {"error": "Cle API manquante. Verifiez NVIDIA_API_KEY dans Render."}\n\n'
        return Response(stream_with_context(err_gen()), mimetype="text/event-stream")

    data = request.get_json()
    messages = data.get("messages", [])
    mode = data.get("mode", "moyen")
    config = data.get("config", {})

    # Récupère la config du mode
    mode_cfg = MODES.get(mode, MODES["moyen"])

    # Personnalisation utilisateur
    ai_name = config.get("aiName", "NOVA")
    personality = config.get("personality", "Tu es un assistant IA utile et bienveillant.")
    user_temp = config.get("temperature", mode_cfg["temperature"])
    user_max = config.get("maxTokens", mode_cfg["max_tokens"])

    # Limites : on prend le plus petit entre le mode et le user
    temperature = float(user_temp)
    max_tokens = min(int(user_max), mode_cfg["max_tokens"])

    # Date
    now = datetime.now()
    date_str = f"{JOURS[now.weekday()]} {now.day} {MOIS[now.month - 1]} {now.year}"

    # Prompt système
    system_prompt = (
        f"Tu t'appelles {ai_name}. {personality} "
        f"Tu reponds TOUJOURS en francais, sans exception. "
        f"{mode_cfg['suffix']} "
        f"La date actuelle est le {date_str}."
    )

    payload_base = {
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    def generate():
        yield b": keepalive\n\n"

        last_error = None
        for model in MODELS:
            for attempt in range(1, 4):  # 3 tentatives par modèle
                try:
                    print(f"[NOVA] Essai {attempt}/3 - {model}", flush=True)
                    payload = {**payload_base, "model": model}

                    r = requests.post(
                        NVIDIA_URL,
                        json=payload,
                        headers=headers,
                        stream=True,
                        timeout=(15, 120)
                    )

                    if r.status_code == 200:
                        print(f"[NOVA] OK avec {model}", flush=True)
                        has_content = False
                        for line in r.iter_lines():
                            if line:
                                has_content = True
                                yield line + b"\n"
                        if has_content:
                            return
                        last_error = "Réponse vide"
                        continue

                    elif r.status_code == 410:
                        print(f"[NOVA] {model} retiré (410)", flush=True)
                        last_error = f"Modèle {model} indisponible"
                        break

                    elif r.status_code in (401, 403):
                        err = r.text[:150].replace('"', "'").replace("\n", " ")
                        yield f'data: {{"error": "Authentification échouée ({r.status_code})."}}\n\n'.encode()
                        return

                    elif r.status_code == 429:
                        print(f"[NOVA] Rate limit (429)", flush=True)
                        last_error = "Limite atteinte, réessayez dans quelques secondes"
                        import time
                        time.sleep(2)
                        continue

                    else:
                        err = r.text[:150].replace('"', "'").replace("\n", " ")
                        print(f"[NOVA] Erreur {r.status_code}: {err}", flush=True)
                        last_error = f"Erreur {r.status_code}"
                        continue

                except requests.exceptions.Timeout:
                    print(f"[NOVA] Timeout avec {model}", flush=True)
                    last_error = "Timeout du serveur IA"
                    continue
                except Exception as e:
                    print(f"[NOVA] Exception: {str(e)[:100]}", flush=True)
                    last_error = str(e)[:100]
                    continue

        err_msg = (last_error or "Service indisponible").replace('"', "'")
        yield f'data: {{"error": "Service IA temporairement indisponible. {err_msg}"}}\n\n'.encode()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.route("/")
def index():
    return send_from_directory(".", "nova.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    key = get_key()
    print(f"[NOVA] Clé API : {'OK (' + str(len(key)) + ' chars)' if key else 'MANQUANTE'}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
