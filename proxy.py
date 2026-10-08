import os
import json
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


_k = get_key()
print("=" * 60, flush=True)
print(f"[NOVA] Cle API : {'OK (' + str(len(_k)) + ' chars)' if _k else 'MANQUANTE'}", flush=True)
print("=" * 60, flush=True)


@app.route("/debug")
def debug():
    key = get_key()
    return {
        "key_present": bool(key),
        "key_length": len(key),
        "key_valid": key.startswith("nvapi-") and len(key) > 50,
    }


# ============================================================
# FONCTION : essai streaming sur un modèle
# Retourne (status, generator)
# status : "ok" | "fail" | "auth" | "empty"
# ============================================================
def stream_from_model(model, payload, headers):
    """Générateur qui yield les lignes SSE. Compte les caractères reçus."""
    payload = {**payload, "model": model, "stream": True}

    try:
        print(f"[NOVA] Streaming: {model}", flush=True)
        r = requests.post(NVIDIA_URL, json=payload, headers=headers, stream=True, timeout=(15, 180))

        if r.status_code in (401, 403):
            print(f"[NOVA] AUTH erreur {r.status_code}", flush=True)
            return "auth", None
        if r.status_code == 410:
            print(f"[NOVA] {model} retiré (410)", flush=True)
            return "fail", None
        if r.status_code == 429:
            print(f"[NOVA] Rate limit {model}", flush=True)
            time.sleep(2)
            return "fail", None
        if r.status_code != 200:
            print(f"[NOVA] Erreur {r.status_code}: {r.text[:100]}", flush=True)
            return "fail", None

        # Fonction génératrice interne
        def gen():
            char_count = 0
            for line in r.iter_lines():
                if not line:
                    continue
                yield line + b"\n"
                # Compter les caractères de contenu reels
                try:
                    l = line.decode('utf-8', errors='ignore')
                    if l.startswith('data:'):
                        payload_str = l[5:].strip()
                        if payload_str and payload_str != '[DONE]':
                            obj = json.loads(payload_str)
                            delta = obj.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                char_count += len(content)
                except Exception:
                    pass

            print(f"[NOVA] Fin stream {model} : {char_count} chars", flush=True)
            # Marqueur de fin
            if char_count == 0:
                # On ne peut plus yield, mais on a déjà yieldé du vide
                pass

        return "ok", gen()

    except requests.exceptions.Timeout:
        print(f"[NOVA] Timeout {model}", flush=True)
        return "fail", None
    except Exception as e:
        print(f"[NOVA] Exception {model}: {str(e)[:100]}", flush=True)
        return "fail", None


# ============================================================
# FONCTION : essai NON-streaming
# Retourne (status, text_content)
# ============================================================
def call_model_sync(model, payload, headers):
    payload = {**payload, "model": model, "stream": False}

    try:
        print(f"[NOVA] Sync: {model}", flush=True)
        r = requests.post(NVIDIA_URL, json=payload, headers=headers, timeout=(15, 120))

        if r.status_code == 200:
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content and content.strip():
                print(f"[NOVA] Sync OK {model} : {len(content)} chars", flush=True)
                return "ok", content
            return "fail", None
        elif r.status_code in (401, 403):
            return "auth", None
        else:
            print(f"[NOVA] Sync erreur {r.status_code}", flush=True)
            return "fail", None
    except Exception as e:
        print(f"[NOVA] Sync exception: {str(e)[:100]}", flush=True)
        return "fail", None


# ============================================================
# ROUTE CHAT
# ============================================================
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
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    def generate():
        yield b": keepalive\n\n"

        # ============ ÉTAPE 1 : Essayer le streaming ============
        for model in MODELS:
            status, gen = stream_from_model(model, payload_base, headers)

            if status == "auth":
                yield b'data: {"error": "Cle API invalide."}\n\n'
                return

            if status == "ok" and gen:
                # On streame, mais on doit vérifier qu'on a bien eu du contenu
                any_content = False
                buffer = []
                for chunk in gen:
                    buffer.append(chunk)
                    try:
                        l = chunk.decode('utf-8', errors='ignore')
                        if '"content":"' in l.replace(' ', ''):
                            # Contient un content non vide
                            if 'null' not in l.split('"content":')[1][:10]:
                                any_content = True
                    except:
                        pass

                # Envoyer tout
                for chunk in buffer:
                    yield chunk

                if any_content:
                    print(f"[NOVA] SUCCESS streaming {model}", flush=True)
                    return
                else:
                    print(f"[NOVA] Streaming vide {model}, on essaie suivant", flush=True)
                    continue

        # ============ ÉTAPE 2 : Fallback non-streaming ============
        print("[NOVA] === Fallback non-streaming ===", flush=True)
        for model in MODELS:
            status, content = call_model_sync(model, payload_base, headers)

            if status == "auth":
                yield b'data: {"error": "Cle API invalide."}\n\n'
                return

            if status == "ok" and content:
                # Simuler le streaming en envoyant mot par mot
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
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                yield b"data: [DONE]\n\n"
                print(f"[NOVA] SUCCESS fallback {model}", flush=True)
                return

        # ============ ÉTAPE 3 : Tout a échoué ============
        yield b'data: {"error": "Service IA indisponible. Reessayez dans quelques secondes."}\n\n'

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


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
