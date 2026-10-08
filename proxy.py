import os
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

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


print("=" * 60, flush=True)
_key = get_key()
print(f"[NOVA] Cle API : {'OK (' + str(len(_key)) + ' chars)' if _key else 'MANQUANTE'}", flush=True)
print("=" * 60, flush=True)


@app.route("/debug")
def debug():
    key = get_key()
    return {
        "key_present": bool(key),
        "key_length": len(key),
        "key_valid": key.startswith("nvapi-") and len(key) > 50,
        "status": "OK" if key.startswith("nvapi-") and len(key) > 50 else "Cle API manquante ou invalide"
    }


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
        yield b": keepalive\n\n"
        last_error = None
        for model in MODELS:
            for attempt in range(1, 4):
                try:
                    print(f"[NOVA] Tentative {attempt}/3 - {model}", flush=True)
                    payload = {**payload_base, "model": model}
                    r = requests.post(NVIDIA_URL, json=payload, headers=headers, stream=True, timeout=(15, 120))

                    if r.status_code == 200:
                        print(f"[NOVA] OK - {model}", flush=True)
                        has_data = False
                        for line in r.iter_lines():
                            if line:
                                has_data = True
                                yield line + b"\n"
                        if has_data:
                            return
                        last_error = "Reponse vide"
                        continue
                    elif r.status_code == 410:
                        print(f"[NOVA] {model} retire (410)", flush=True)
                        last_error = "Modele indisponible"
                        break
                    elif r.status_code in (401, 403):
                        yield f'data: {{"error": "Cle API invalide."}}\n\n'.encode()
                        return
                    elif r.status_code == 429:
                        import time
                        time.sleep(2)
                        last_error = "Limite atteinte"
                        continue
                    else:
                        err = r.text[:150].replace('"', "'").replace("\n", " ")
                        print(f"[NOVA] Erreur {r.status_code}: {err}", flush=True)
                        last_error = f"HTTP {r.status_code}"
                        continue
                except requests.exceptions.Timeout:
                    print(f"[NOVA] Timeout {model}", flush=True)
                    last_error = "Timeout"
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )


# ============ IMPORTANT : sert nova.html + TOUTES les images ============
@app.route("/")
def index():
    return send_from_directory(".", "nova.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Sert logo.png, text.png, favicon.ico, etc."""
    return send_from_directory(".", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
