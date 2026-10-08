import os
import json
import time
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Liste des modèles à essayer dans l'ordre
MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-nemo-12b-instruct",
    "microsoft/phi-3-medium-128k-instruct",
    "google/gemma-2-27b-it",
]

# Nombre de tentatives par modèle
MAX_RETRIES_PER_MODEL = 3
RETRY_DELAY_SECONDS = 1


def get_key():
    """Récupère la clé de façon fiable à chaque requête."""
    return os.environ.get("NVIDIA_API_KEY", "").strip()


def log(msg):
    """Log avec timestamp."""
    print(f"[NOVA] {msg}", flush=True)


# ============================================================
# ROUTE DEBUG - à ouvrir dans le navigateur pour diagnostic
# ============================================================
@app.route("/debug")
def debug():
    key = get_key()
    info = {
        "key_present": bool(key),
        "key_length": len(key),
        "key_starts_with_nvapi": key.startswith("nvapi-") if key else False,
        "key_valid": bool(key) and len(key) > 50 and key.startswith("nvapi-"),
        "env_var_exists": "NVIDIA_API_KEY" in os.environ,
        "env_var_raw_length": len(os.environ.get("NVIDIA_API_KEY", "")),
        "all_env_var_names": sorted(list(os.environ.keys())),
    }

    # Si la clé est présente, on teste NVIDIA directement
    if info["key_valid"]:
        try:
            r = requests.get(
                "https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10
            )
            info["nvidia_test_status"] = r.status_code
            info["nvidia_test_message"] = r.text[:200]
        except Exception as e:
            info["nvidia_test_status"] = "exception"
            info["nvidia_test_message"] = str(e)[:200]
    else:
        info["nvidia_test_status"] = "non testé (clé invalide)"

    return info


# ============================================================
# FONCTION PRINCIPALE - ESSAIE TOUS LES MODÈLES AVEC RETRY
# ============================================================
def try_all_models(messages, payload_base, headers):
    """
    Essaie tous les modèles, chacun 3 fois.
    Retourne un générateur de streaming dès qu'un modèle répond.
    """
    last_error = None

    for model in MODELS:
        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                log(f"Tentative {attempt}/{MAX_RETRIES_PER_MODEL} - modèle: {model}")

                payload = {**payload_base, "model": model}

                r = requests.post(
                    NVIDIA_URL,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=(15, 120)  # 15s connexion, 120s lecture
                )

                log(f"Status reçu: {r.status_code}")

                if r.status_code == 200:
                    log(f"✅ SUCCÈS avec {model} (tentative {attempt})")
                    # Streamer la réponse
                    got_content = False
                    for line in r.iter_lines():
                        if line:
                            got_content = True
                            yield line + b"\n"
                    if got_content:
                        return  # Terminé avec succès
                    else:
                        log(f"Modèle {model} a renvoyé du vide, on essaie un autre")
                        last_error = "Réponse vide"
                        continue

                elif r.status_code == 429:
                    log(f"⚠️ Rate limit (429), attente {RETRY_DELAY_SECONDS}s...")
                    last_error = f"Rate limit sur {model}"
                    time.sleep(RETRY_DELAY_SECONDS * 2)
                    continue

                elif r.status_code == 410:
                    log(f"❌ Modèle {model} retiré (410), on passe au suivant")
                    last_error = f"Modèle {model} indisponible"
                    break  # Passer au modèle suivant immédiatement

                elif r.status_code in (401, 403):
                    log(f"❌ Authentification échouée ({r.status_code})")
                    err_text = r.text[:200].replace('"', "'").replace("\n", " ")
                    yield f'data: {{"error": "Authentification échouée. Vérifiez la clé API."}}\n\n'.encode("utf-8")
                    return

                else:
                    err_text = r.text[:200].replace('"', "'").replace("\n", " ")
                    log(f"❌ Erreur {r.status_code}: {err_text}")
                    last_error = f"HTTP {r.status_code} sur {model}"
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue

            except requests.exceptions.Timeout:
                log(f"⏱️ Timeout avec {model} (tentative {attempt})")
                last_error = f"Timeout sur {model}"
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            except requests.exceptions.ConnectionError as e:
                log(f"🔌 Erreur connexion avec {model}: {str(e)[:100]}")
                last_error = f"Erreur connexion"
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            except Exception as e:
                log(f"❌ Exception avec {model}: {str(e)[:100]}")
                last_error = str(e)[:100]
                time.sleep(RETRY_DELAY_SECONDS)
                continue

    # Si tous les modèles ont échoué
    log(f"❌ TOUS LES MODÈLES ONT ÉCHOUÉ. Dernière erreur: {last_error}")
    err_msg = (last_error or "Service indisponible").replace('"', "'")
    yield f'data: {{"error": "Service IA temporairement indisponible. Détails: {err_msg}"}}\n\n'.encode("utf-8")


# ============================================================
# ROUTE CHAT
# ============================================================
@app.route("/api/chat", methods=["POST"])
def chat():
    key = get_key()

    log(f"Requête chat reçue. Clé valide: {bool(key) and len(key) > 50}")

    if not key or len(key) < 50 or not key.startswith("nvapi-"):
        def err_gen():
            msg = f"Clé API invalide ou manquante (longueur={len(key)}). Vérifiez la variable NVIDIA_API_KEY dans Render."
            yield f'data: {{"error": "{msg}"}}\n\n'.encode("utf-8")
        return Response(stream_with_context(err_gen()), mimetype="text/event-stream")

    try:
        data = request.get_json()
    except Exception as e:
        def err_gen():
            yield f'data: {{"error": "Requête invalide: {str(e)[:100]}"}}\n\n'.encode("utf-8")
        return Response(stream_with_context(err_gen()), mimetype="text/event-stream")

    messages = data.get("messages", [])
    config = data.get("config", {})
    mode = data.get("mode", "moyen")

    ai_name = config.get("aiName", "NOVA")
    personality = config.get("personality", "Tu es un assistant IA utile et bienveillant.")
    temperature = float(config.get("temperature", 0.7))

    mode_limits = {"faible": 1000, "moyen": 2500, "max": 4000}
    max_tokens = min(int(config.get("maxTokens", 2500)), mode_limits.get(mode, 2500))

    now = datetime.now()
    date_str = f"{now.day}/{now.month}/{now.year}"
    system_prompt = (
        f"Tu t'appelles {ai_name}. {personality} "
        f"Tu reponds TOUJOURS en francais, sans exception. Nous sommes le {date_str}."
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
        # Envoi keepalive immédiat pour éviter les timeouts navigateur
        yield b": keepalive\n\n"
        # Utiliser notre fonction robuste
        yield from try_all_models(messages, payload_base, headers)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.route("/")
def index():
    return send_from_directory(".", "nova.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    key = get_key()
    log(f"Démarrage local - Clé présente: {bool(key)} (longueur: {len(key)})")
    app.run(host="0.0.0.0", port=port, debug=False)
