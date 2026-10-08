import os
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests
import time

app = Flask(__name__)
CORS(app)

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-nemo-12b-instruct",
]

MODES = {
    "faible": {"max_tokens": 1000, "temperature": 0.3, "suffix": " Reponds de facon concise."},
    "moyen":  {"max_tokens": 2500, "temperature": 0.7, "suffix": " Sois clair et equilibre."},
    "max":    {"max_tokens": 4000, "temperature": 1.0, "suffix": " Analyse en profondeur."}
}

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "fevrier", "mars", "avril", "mai", "juin",
        "juillet", "aout", "septembre", "octobre", "novembre", "decembre"]


def get_key():
    return os.environ.get("NVIDIA_API_KEY", "").strip()


_key_check = get_key()
print("=" * 60, flush=True)
print(f"[NOVA] Cle API : {'OK (' + str(len(_key_check)) + ' chars)' if _key_check else 'MANQUANTE'}", flush=True)
print("=" * 60, flush=True)


@app.route("/debug")
def debug():
    key = get_key()
    return {
        "key_present": bool(key),
        "key_length": len(key),
        "key_valid": key.startswith("nvapi-") and len(key) > 50,
        "status": "OK" if key.startswith("nvapi-") and len(key) > 50 else "Cle API manquante"
    }


def try_model(model, payload_base, headers):
    """
    Essaie un modèle en streaming.
    Retourne (success, generator_or_None)
    """
    payload = {**payload_base, "model": model}

    try:
        print(f"[NOVA] Essai streaming: {model}", flush=True)
        r = requests.post(
            NVIDIA_URL,
            json=payload,
            headers=headers,
            stream=True,
            timeout=(15, 180)  # 15s connexion, 180s lecture
        )

        if r.status_code == 200:
            # Collecter les lignes SSE
            lines_buffer = []
            has_content = False
            for line in r.iter_lines():
                if line:
                    lines_buffer.append(line + b"\n")
                    # Vérifie si cette ligne contient du contenu
                    if b'"content":"' in line or b'"content": "' in line:
                        has_content = True

            if has_content:
                print(f"[NOVA] SUCCESS streaming: {model} ({len(lines_buffer)} lignes)", flush=True)
                return True, lines_buffer
            else:
                print(f"[NOVA] Streaming vide pour {model}", flush=True)
                return False, None

        elif r.status_code == 410:
            print(f"[NOVA] {model} retiré (410)", flush=True)
            return False, None

        elif r.status_code in (401, 403):
            print(f"[NOVA] Auth error {r.status_code}", flush=True)
            return "AUTH_ERROR", None

        elif r.status_code == 429:
            print(f"[NOVA] Rate limit {model}", flush=True)
            time.sleep(2)
            return False, None

        else:
            err = r.text[:200]
            print(f"[NOVA] Erreur {r.status_code}: {err}", flush=True)
            return False, None

    except requests.exceptions.Timeout:
        print(f"[NOVA] Timeout streaming: {model}", flush=True)
        return False, None
    except Exception as e:
        print(f"[NOVA] Exception streaming: {str(e)[:100]}", flush=True)
        return False, None


def try_model_non_streaming(model, payload_base, headers):
    """Fallback : essaie en non-streaming si le streaming échoue."""
    payload = {**payload_base, "model": model, "stream": False}

    try:
        print(f"[NOVA] Essai non-streaming: {model}", flush=True)
        r = requests.post(NVIDIA_URL, json=payload, headers=headers, timeout=(15, 120))

        if r.status_code == 200:
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                print(f"[NOVA] SUCCESS non-streaming: {model}", flush=True)
                return True, content
            else:
                print(f"[NOVA] Non-streaming vide: {model}", flush=True)
                return False, None
        else:
            print(f"[NOVA] Non-streaming erreur {r.status_code}", flush=True)
            return False, None
    except Exception as e:
        print(f"[NOVA] Exception non-streaming: {str(e)[:100]}", flush=True)
        return False, None


@app.route("/api/chat", methods=["POST"])
def chat():
    key = get_key()
    if not key or len(key) < 50:
        def err_gen():
            yield b'data: {"error": "Cle API manquante."}\n\n'
        return Response(stream_with_context(err_gen()), mimetype="text/event-stream")

    data = request.get_json()
    messages = data.get("messages", [])
    mode = data.get("mode", "moyen")
    config = data.get("config", {})

    mode_cfg = MODES.get(mode, MODES["moyen"])
    ai_name = config.get("aiName", "NOVA")
    personality = config.get("personality", "Tu es un assistant IA utile et bienveillant.")
    temperature = float(config.get("temperature", mode_cfg["temperature"]))
    max_tokens = min(int(config.get("maxTokens", mode_cfg["max_tokens"])), mode_cfg["max_tokens"])

    now = datetime.now()
    date_str = f"{JOURS[now.weekday()]} {now.day} {MOIS[now.month - 1]} {now.year}"

    system_prompt = (
        f"Tu t'appelles {ai_name}. {personality} "
        f"Tu reponds TOUJOURS en francais, sans exception. "
        f"{mode_cfg['suffix']} La date actuelle est le {date_str}."
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
        # Keepalive immédiat
        yield b": keepalive\n\n"

        # ÉTAPE 1 : Streaming (essaie 2 fois par modèle)
        for model in MODELS:
            for attempt in range(1, 3):
                print(f"[NOVA] Tentative streaming {attempt}/2 - {model}", flush=True)
                success, result = try_model(model, payload_base, headers)

                if success == "AUTH_ERROR":
                    yield b'data: {"error": "Cle API invalide."}\n\n'
                    return

                if success and result:
                    for line in result:
                        yield line
                    return

                # Si échec, on essaie le suivant
                time.sleep(1)

        # ÉTAPE 2 : Non-streaming en fallback (si tout le streaming a échoué)
        print(f"[NOVA] === Fallback non-streaming ===", flush=True)
        for model in MODELS:
            success, content = try_model_non_streaming(model, payload_base, headers)
            if success and content:
                # Convertir en format SSE pour le frontend
                import json as _json
                # Découper le contenu en petits morceaux pour simuler le streaming
                words = content.split(' ')
                for i, word in enumerate(words):
                    delta = word + (' ' if i < len(words) - 1 else '')
                    chunk = {
                        "choices": [{
                            "delta": {"content": delta},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {_json.dumps(chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                yield b"data: [DONE]\n\n"
                return

        # ÉTAPE 3 : Si vraiment rien ne marche
        yield b'data: {"error": "Service IA indisponible. Reessayez dans quelques secondes."}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )


@app.route("/")
def index():
    return send_from_directory(".", "nova.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
