import os
import uuid
import threading
import time
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()

# ====== CHAT ======
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_CHAT = os.environ.get("NVIDIA_MODEL_CHAT", "nvidia/nemotron-3.5-lightning-30b-a3b")

# ====== IMAGE (FLUX.2-klein-4B) ======
NVIDIA_IMAGE_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b"

MODES = {
    "faible": {"max_tokens": 2000, "temperature": 0.3, "suffix": " Reponds de facon concise."},
    "moyen":  {"max_tokens": 4000, "temperature": 0.7, "suffix": " Sois clair et equilibre."},
    "max":    {"max_tokens": 8000, "temperature": 1.0, "suffix": " Analyse en profondeur."}
}

CONNECT_TIMEOUT = 15
READ_TIMEOUT = 120

# Stockage en mémoire pour les jobs d'images
IMAGE_JOBS = {}

# ============================================================
# ROUTE CHAT
# ============================================================
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
        "model": MODEL_CHAT,
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
            with requests.post(NVIDIA_CHAT_URL, json=payload, headers=headers,
                               stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
                if r.status_code != 200:
                    err = r.text[:200].replace('"', "'").replace("\n", " ")
                    yield f'data: {{"error": "API {r.status_code}: {err}"}}\n\n'.encode()
                    return
                for line in r.iter_lines():
                    if line:
                        yield line + b"\n"
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

# ============================================================
# GÉNÉRATION D'IMAGES - Job + Polling (compatible free tier)
# ============================================================

def _run_image_job(job_id, prompt):
    """Fonction exécutée en arrière-plan pour la génération d'image."""
    try:
        IMAGE_JOBS[job_id]["status"] = "generating"
        IMAGE_JOBS[job_id]["progress"] = 10

        # Payload spécifique pour FLUX.2-klein-4B
        payload = {
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 0,
            "guidance_scale": 1.0
        }
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        IMAGE_JOBS[job_id]["progress"] = 30

        # Timeout long autorisé car en arrière-plan
        r = requests.post(NVIDIA_IMAGE_URL, json=payload, headers=headers,
                          timeout=(20, 300))

        IMAGE_JOBS[job_id]["progress"] = 85

        if r.status_code != 200:
            err_full = r.text[:500]
            print(f"[NOVA][ERREUR IMAGE FLUX.2] {r.status_code} -> {err_full}", flush=True)
            IMAGE_JOBS[job_id]["status"] = "error"
            IMAGE_JOBS[job_id]["error"] = f"API {r.status_code}: {err_full[:200]}"
            return

        result = r.json()
        image_url = None

        # FLUX.2-klein-4B renvoie généralement un champ "image" ou "artifacts"
        if result.get("image"):
            img = result["image"]
            image_url = img if img.startswith("data:") else "data:image/png;base64," + img
        else:
            artifacts = result.get("artifacts") or []
            if artifacts and isinstance(artifacts[0], dict):
                art = artifacts[0]
                if art.get("base64"):
                    image_url = "data:image/png;base64," + art["base64"]
                elif art.get("url"):
                    image_url = art["url"]

        if not image_url:
            print(f"[NOVA][IMAGE FORMAT INCONNU] {str(result)[:300]}", flush=True)
            IMAGE_JOBS[job_id]["status"] = "error"
            IMAGE_JOBS[job_id]["error"] = "Format de réponse inconnu."
            return

        IMAGE_JOBS[job_id]["status"] = "done"
        IMAGE_JOBS[job_id]["progress"] = 100
        IMAGE_JOBS[job_id]["image"] = image_url
        print(f"[NOVA][IMAGE OK] job {job_id} termine", flush=True)

    except Exception as e:
        print(f"[NOVA][EXCEPTION IMAGE] {e}", flush=True)
        IMAGE_JOBS[job_id]["status"] = "error"
        IMAGE_JOBS[job_id]["error"] = str(e)[:200]


@app.route("/api/image/start", methods=["POST"])
def image_start():
    """Démarre un job de génération d'image et retourne un ID immédiatement."""
    if not NVIDIA_API_KEY:
        return {"error": "Cle API manquante."}, 500

    data = request.get_json() or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return {"error": "Prompt vide."}, 400

    job_id = str(uuid.uuid4())
    IMAGE_JOBS[job_id] = {
        "status": "pending",
        "progress": 0,
        "image": None,
        "error": None,
        "created": time.time(),
    }

    # Lance la requête en arrière-plan
    threading.Thread(target=_run_image_job, args=(job_id, prompt), daemon=True).start()

    # Nettoyage des jobs de plus de 15 minutes
    now = time.time()
    for k in list(IMAGE_JOBS.keys()):
        if now - IMAGE_JOBS[k].get("created", now) > 900:
            del IMAGE_JOBS[k]

    return {"job_id": job_id}


@app.route("/api/image/status/<job_id>")
def image_status(job_id):
    """Retourne l'état actuel du job."""
    job = IMAGE_JOBS.get(job_id)
    if not job:
        return {"error": "Job introuvable ou expiré."}, 404
    return {
        "status": job["status"],
        "progress": job["progress"],
        "image": job["image"],
        "error": job["error"],
    }

# ============================================================
# ROUTES DE DEBUG ET FICHIERS STATIQUES
# ============================================================
@app.route("/debug")
def debug():
    return {
        "key_present": bool(NVIDIA_API_KEY),
        "key_length": len(NVIDIA_API_KEY),
        "key_valid": NVIDIA_API_KEY.startswith("nvapi-") and len(NVIDIA_API_KEY) > 50,
        "chat_model": MODEL_CHAT,
        "image_url": NVIDIA_IMAGE_URL,
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
    print(f"[NOVA] Image URL : {NVIDIA_IMAGE_URL}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
