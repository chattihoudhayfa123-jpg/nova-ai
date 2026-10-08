import os
import io
import uuid
import base64
import threading
import time
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory, jsonify
from flask_cors import CORS
import requests

try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_URL = f"{NVIDIA_BASE}/chat/completions"
NVIDIA_IMAGE_BASE = "https://ai.api.nvidia.com/v1/genai"

# ============================================================
# CATALOGUE DES MODÈLES (TEXTE)
# ============================================================
TEXT_MODELS = {
    "nvidia/nemotron-3.5-lightning-30b-a3b": {
        "id": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "name": "Nemotron 3.5 Lightning",
        "provider": "NVIDIA",
        "description": "Modèle léger et ultra-rapide, idéal pour les conversations fluides.",
        "context": "32K tokens",
        "queue": "Très faible",
        "speed": "★★★★★ (Ultra rapide)",
        "efficiency": "★★★★☆",
        "best_for": "Chat quotidien, réponses rapides",
        "default": True,
    },
    "nvidia/llama-3.1-nemotron-70b-instruct": {
        "id": "nvidia/llama-3.1-nemotron-70b-instruct",
        "name": "Llama 3.1 Nemotron 70B",
        "provider": "NVIDIA / Meta",
        "description": "Grand modèle de raisonnement, excellent pour les tâches complexes.",
        "context": "128K tokens",
        "queue": "Modérée",
        "speed": "★★★☆☆ (Moyen)",
        "efficiency": "★★★★★",
        "best_for": "Analyse, code, raisonnement complexe",
        "default": False,
    },
    "meta/llama-3.1-8b-instruct": {
        "id": "meta/llama-3.1-8b-instruct",
        "name": "Llama 3.1 8B Instruct",
        "provider": "Meta",
        "description": "Petit modèle très rapide, bon compromis légèreté/qualité.",
        "context": "128K tokens",
        "queue": "Très faible",
        "speed": "★★★★★ (Très rapide)",
        "efficiency": "★★★☆☆",
        "best_for": "Réponses courtes, tâches simples",
        "default": False,
    },
    "meta/llama-3.2-11b-vision-instruct": {
        "id": "meta/llama-3.2-11b-vision-instruct",
        "name": "Llama 3.2 11B Vision",
        "provider": "Meta",
        "description": "Modèle multimodal qui comprend les images et le texte.",
        "context": "128K tokens",
        "queue": "Faible",
        "speed": "★★★★☆",
        "efficiency": "★★★★☆",
        "best_for": "Analyse d'images, description visuelle",
        "default": False,
    },
    "deepseek-ai/deepseek-r1-distill-llama-8b": {
        "id": "deepseek-ai/deepseek-r1-distill-llama-8b",
        "name": "DeepSeek R1 Distill 8B",
        "provider": "DeepSeek",
        "description": "Spécialisé dans le raisonnement étape par étape.",
        "context": "64K tokens",
        "queue": "Modérée",
        "speed": "★★★☆☆",
        "efficiency": "★★★★☆",
        "best_for": "Maths, logique, réflexion structurée",
        "default": False,
    },
}

# ============================================================
# CATALOGUE DES MODÈLES (IMAGE)
# ============================================================
IMAGE_MODELS = {
    "black-forest-labs/flux.2-klein-4b": {
        "id": "black-forest-labs/flux.2-klein-4b",
        "name": "FLUX.2 Klein 4B",
        "provider": "Black Forest Labs",
        "description": "Modèle distillé ultra-rapide (4 étapes), génération en quelques secondes.",
        "steps": 4,
        "queue": "Faible",
        "speed": "★★★★★ (Éclair)",
        "efficiency": "★★★★☆",
        "best_for": "Génération rapide, style artistique",
        "endpoint": f"{NVIDIA_IMAGE_BASE}/black-forest-labs/flux.2-klein-4b",
        "payload_format": "flux",
        "default": True,
    },
    "black-forest-labs/flux.1-schnell": {
        "id": "black-forest-labs/flux.1-schnell",
        "name": "FLUX.1 Schnell",
        "provider": "Black Forest Labs",
        "description": "Version rapide de FLUX.1, très bonne qualité en 4 étapes.",
        "steps": 4,
        "queue": "Élevée (très demandé)",
        "speed": "★★★★☆",
        "efficiency": "★★★★☆",
        "best_for": "Qualité rapide, style photo",
        "endpoint": f"{NVIDIA_IMAGE_BASE}/black-forest-labs/flux.1-schnell",
        "payload_format": "flux",
        "default": False,
    },
    "black-forest-labs/flux.1-dev": {
        "id": "black-forest-labs/flux.1-dev",
        "name": "FLUX.1 Dev",
        "provider": "Black Forest Labs",
        "description": "Modèle de haute qualité, meilleure fidélité au prompt.",
        "steps": 50,
        "queue": "Très élevée",
        "speed": "★★☆☆☆ (Lent)",
        "efficiency": "★★★★★",
        "best_for": "Qualité maximale, prompts complexes",
        "endpoint": f"{NVIDIA_IMAGE_BASE}/black-forest-labs/flux.1-dev",
        "payload_format": "flux",
        "default": False,
    },
    "stabilityai/stable-diffusion-3.5-large": {
        "id": "stabilityai/stable-diffusion-3.5-large",
        "name": "Stable Diffusion 3.5 Large",
        "provider": "Stability AI",
        "description": "Polyvalent, très bon rendu des détails et des textes.",
        "steps": 30,
        "queue": "Modérée",
        "speed": "★★★☆☆",
        "efficiency": "★★★★★",
        "best_for": "Illustrations détaillées, texte dans l'image",
        "endpoint": f"{NVIDIA_IMAGE_BASE}/stabilityai/stable-diffusion-3.5-large",
        "payload_format": "sdxl",
        "default": False,
    },
    "stabilityai/stable-diffusion-xl": {
        "id": "stabilityai/stable-diffusion-xl",
        "name": "Stable Diffusion XL",
        "provider": "Stability AI",
        "description": "Modèle classique et fiable, bon équilibre vitesse/qualité.",
        "steps": 30,
        "queue": "Faible",
        "speed": "★★★★☆",
        "efficiency": "★★★★☆",
        "best_for": "Généraliste, portraits, paysages",
        "endpoint": f"{NVIDIA_IMAGE_BASE}/stabilityai/stable-diffusion-xl",
        "payload_format": "sdxl",
        "default": False,
    },
    "qwen/qwen-image": {
        "id": "qwen/qwen-image",
        "name": "Qwen Image",
        "provider": "Alibaba Qwen",
        "description": "Spécialisé dans le rendu de texte multilingue dans l'image.",
        "steps": 20,
        "queue": "Faible",
        "speed": "★★★★☆",
        "efficiency": "★★★★☆",
        "best_for": "Affiches, logos, texte dans l'image",
        "endpoint": f"{NVIDIA_IMAGE_BASE}/qwen/qwen-image",
        "payload_format": "flux",
        "default": False,
    },
}

# Modèle actif (par défaut)
ACTIVE_TEXT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
ACTIVE_IMAGE_MODEL = "black-forest-labs/flux.2-klein-4b"

# ============================================================
# UTILITAIRES
# ============================================================
def _truncate_prompt(prompt, max_len=780):
    prompt = (prompt or "").strip()
    if len(prompt) <= max_len:
        return prompt
    cut = prompt[:max_len]
    for sep in ['. ', '.\n', '! ', '? ', '; ', ', ', '\n', ' ']:
        idx = cut.rfind(sep)
        if idx > max_len * 0.5:
            return cut[:idx + len(sep)].rstrip() + '...'
    return cut.rstrip() + '...'


def _describe_image(data_url, max_words=250):
    if not NVIDIA_API_KEY or not data_url:
        return ""
    try:
        prompt = (
            "Décris cette image de façon purement visuelle, neutre et positive, "
            "en français, comme pour un prompt de génération d'image artistique.\n\n"
            "Décris UNIQUEMENT :\n"
            "- Les formes et objets visibles\n"
            "- Les couleurs dominantes et leurs nuances\n"
            "- Le style visuel (photo, illustration, peinture, 3D...)\n"
            "- L'ambiance générale\n"
            "- Les éléments de décor\n\n"
            f"Maximum {max_words} mots. Sois descriptif mais neutre. "
            "Ne mentionne NI personne, NI visage, NI émotion, NI marque, NI texte. "
            "Réponds uniquement par la description."
        )
        payload = {
            "model": "meta/llama-3.2-11b-vision-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            }],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 500,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        r = requests.post(NVIDIA_CHAT_URL, json=payload, headers=headers,
                          timeout=(15, 90))
        if r.status_code != 200:
            print(f"[NOVA][VISION ERR] {r.status_code} -> {r.text[:300]}", flush=True)
            return ""
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        description = (content or "").strip()
        print(f"[NOVA][VISION OK] len={len(description)} chars", flush=True)
        return description
    except Exception as e:
        print(f"[NOVA][VISION EXCEPTION] {e}", flush=True)
        return ""


# ============================================================
# ROUTE : CATALOGUE DES MODÈLES
# ============================================================
@app.route("/api/models", methods=["GET"])
def get_models():
    """Retourne la liste des modèles texte et image disponibles."""
    return jsonify({
        "text_models": list(TEXT_MODELS.values()),
        "image_models": list(IMAGE_MODELS.values()),
        "active_text_model": ACTIVE_TEXT_MODEL,
        "active_image_model": ACTIVE_IMAGE_MODEL,
    })


@app.route("/api/models/select", methods=["POST"])
def select_model():
    """Change le modèle actif."""
    global ACTIVE_TEXT_MODEL, ACTIVE_IMAGE_MODEL
    data = request.get_json() or {}
    model_type = data.get("type", "text")
    model_id = data.get("model_id", "")

    if model_type == "text":
        if model_id not in TEXT_MODELS:
            return {"error": "Modèle texte inconnu."}, 400
        ACTIVE_TEXT_MODEL = model_id
        print(f"[NOVA] Modèle texte actif : {ACTIVE_TEXT_MODEL}", flush=True)
        return {"ok": True, "active": ACTIVE_TEXT_MODEL}
    elif model_type == "image":
        if model_id not in IMAGE_MODELS:
            return {"error": "Modèle image inconnu."}, 400
        ACTIVE_IMAGE_MODEL = model_id
        print(f"[NOVA] Modèle image actif : {ACTIVE_IMAGE_MODEL}", flush=True)
        return {"ok": True, "active": ACTIVE_IMAGE_MODEL}
    return {"error": "Type inconnu."}, 400


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
    attachments = data.get("attachments", [])

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

    has_image = any(a.get("type") == "image" for a in attachments)
    model = "meta/llama-3.2-11b-vision-instruct" if has_image else ACTIVE_TEXT_MODEL

    trimmed = messages[-20:] if len(messages) > 20 else messages[:]
    final_messages = [{"role": "system", "content": system_prompt}]

    for idx, m in enumerate(trimmed):
        is_last_user = (idx == len(trimmed) - 1 and m.get("role") == "user")

        if is_last_user and attachments:
            content_parts = []
            if m.get("content"):
                content_parts.append({"type": "text", "text": m["content"]})

            for att in attachments:
                if att.get("type") == "text":
                    content_parts.append({
                        "type": "text",
                        "text": (
                            f"\n\n--- Fichier joint : {att.get('filename','document')} ---\n"
                            f"{att.get('content','')}\n--- Fin du fichier ---\n"
                        )
                    })
                elif att.get("type") == "image":
                    desc = att.get("description") or att.get("_description")
                    if desc:
                        content_parts.append({
                            "type": "text",
                            "text": f"\n\n[Description de l'image : {desc}]\n"
                        })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": att.get("data_url", "")}
                    })

            final_messages.append({"role": "user", "content": content_parts})
        else:
            final_messages.append({
                "role": m.get("role", "user"),
                "content": m.get("content", "")
            })

    payload = {
        "model": model,
        "messages": final_messages,
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
                               stream=True, timeout=(15, 120)) as r:
                if r.status_code != 200:
                    err = r.text[:300].replace('"', "'").replace("\n", " ")
                    print(f"[NOVA][ERREUR CHAT] {r.status_code} -> {err}", flush=True)
                    yield f'data: {{"error": "API {r.status_code}: {err}"}}\n\n'.encode()
                    return
                for line in r.iter_lines():
                    if line:
                        yield line + b"\n"
        except Exception as e:
            print(f"[NOVA][EXCEPTION CHAT] {e}", flush=True)
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
# UPLOAD
# ============================================================
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".css", ".yml", ".yaml", ".toml",
    ".ini", ".cfg", ".log", ".sql", ".sh", ".bat", ".java", ".c", ".cpp",
    ".h", ".hpp", ".rs", ".go", ".rb", ".php", ".swift", ".kt", ".r",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return {"error": "Aucun fichier reçu."}, 400

    f = request.files["file"]
    if not f.filename:
        return {"error": "Nom de fichier vide."}, 400

    filename = f.filename
    ext = os.path.splitext(filename)[1].lower()
    try:
        file_bytes = f.read()
        size = len(file_bytes)
    except Exception as e:
        return {"error": f"Lecture impossible : {e}"}, 500

    if ext == ".pdf":
        if not PYPDF_OK:
            return {"error": "Support PDF indisponible (pypdf non installé)."}, 500
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages[:50]):
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception:
                    pages_text.append("")
            text = "\n".join(pages_text).strip()
            if not text:
                return {"error": "Aucun texte extractible (PDF scanné ?)."}, 400
            return {
                "type": "text", "filename": filename, "size": size,
                "content": text[:50000], "truncated": len(text) > 50000,
            }
        except Exception as e:
            return {"error": f"Erreur PDF : {e}"}, 500

    if ext in TEXT_EXTENSIONS or (ext == "" and size < 100_000):
        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            try:
                text = file_bytes.decode("latin-1", errors="replace")
            except Exception as e:
                return {"error": f"Décodage impossible : {e}"}, 500
        return {
            "type": "text", "filename": filename, "size": size,
            "content": text[:50000], "truncated": len(text) > 50000,
        }

    if ext in IMAGE_EXTENSIONS:
        if size > 8 * 1024 * 1024:
            return {"error": "Image trop volumineuse (max 8 Mo)."}, 400
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
        }.get(ext, "image/png")
        b64 = base64.b64encode(file_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        description = _describe_image(data_url)
        print(f"[NOVA][UPLOAD IMAGE] {filename} | desc = {description[:100]}...", flush=True)
        return {
            "type": "image", "filename": filename, "size": size,
            "mime": mime, "data_url": data_url, "description": description,
        }

    return {"error": f"Type de fichier non supporté : {ext or '?'}"}, 400


# ============================================================
# GÉNÉRATION D'IMAGES
# ============================================================
def _save_image_result(job_id, r):
    result = r.json()
    image_url = None
    artifacts = result.get("artifacts") or []
    if artifacts and isinstance(artifacts[0], dict):
        art = artifacts[0]
        reason = art.get("finishReason") or art.get("finish_reason")
        if reason == "CONTENT_FILTERED":
            IMAGE_JOBS[job_id]["status"] = "error"
            IMAGE_JOBS[job_id]["error"] = (
                "🚫 Description bloquée par le filtre de sécurité NVIDIA. "
                "Essaie un prompt simple et neutre."
            )
            return

    if result.get("image"):
        img = result["image"]
        image_url = img if img.startswith("data:") else "data:image/png;base64," + img
    else:
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
    print(f"[NOVA][IMAGE OK] job {job_id}", flush=True)


def _run_image_job(job_id, user_prompt):
    try:
        final_prompt = user_prompt.strip() if user_prompt.strip() else "a beautiful abstract art piece"
        final_prompt = _truncate_prompt(final_prompt, max_len=780)

        print(f"[NOVA][IMG] prompt ({len(final_prompt)} chars) : {final_prompt[:150]}", flush=True)

        IMAGE_JOBS[job_id]["status"] = "generating"
        IMAGE_JOBS[job_id]["progress"] = 30

        model_info = IMAGE_MODELS.get(ACTIVE_IMAGE_MODEL, IMAGE_MODELS["black-forest-labs/flux.2-klein-4b"])
        endpoint = model_info["endpoint"]
        payload_format = model_info.get("payload_format", "flux")

        if payload_format == "sdxl":
            payload = {
                "text_prompts": [{"text": final_prompt}],
                "cfg_scale": 5.0,
                "seed": 0,
                "steps": model_info.get("steps", 30),
                "width": 1024,
                "height": 1024,
            }
        else:
            payload = {
                "prompt": final_prompt,
                "width": 1024,
                "height": 1024,
                "steps": model_info.get("steps", 4),
                "seed": 0,
            }

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        r = requests.post(endpoint, json=payload, headers=headers, timeout=(20, 300))
        IMAGE_JOBS[job_id]["progress"] = 85

        if r.status_code != 200:
            err_full = r.text[:500]
            print(f"[NOVA][ERREUR IMAGE] {r.status_code} -> {err_full}", flush=True)
            IMAGE_JOBS[job_id]["status"] = "error"
            IMAGE_JOBS[job_id]["error"] = f"API {r.status_code}: {err_full[:200]}"
            return

        _save_image_result(job_id, r)

    except Exception as e:
        print(f"[NOVA][EXCEPTION IMAGE] {e}", flush=True)
        IMAGE_JOBS[job_id]["status"] = "error"
        IMAGE_JOBS[job_id]["error"] = str(e)[:200]


@app.route("/api/image/start", methods=["POST"])
def image_start():
    if not NVIDIA_API_KEY:
        return {"error": "Cle API manquante."}, 500
    data = request.get_json() or {}
    user_prompt = (data.get("prompt") or "").strip()
    reference_description = (data.get("reference_description") or "").strip()

    if not user_prompt and not reference_description:
        return {"error": "Prompt vide."}, 400

    if not user_prompt:
        user_prompt = reference_description[:200]

    print(f"[NOVA][IMG START] prompt='{user_prompt[:80]}'", flush=True)

    job_id = str(uuid.uuid4())
    IMAGE_JOBS[job_id] = {
        "status": "pending", "progress": 0,
        "image": None, "error": None, "created": time.time(),
    }
    threading.Thread(target=_run_image_job, args=(job_id, user_prompt), daemon=True).start()

    now = time.time()
    for k in list(IMAGE_JOBS.keys()):
        if now - IMAGE_JOBS[k].get("created", now) > 900:
            del IMAGE_JOBS[k]

    return {"job_id": job_id}


@app.route("/api/image/status/<job_id>")
def image_status(job_id):
    job = IMAGE_JOBS.get(job_id)
    if not job:
        return {"error": "Job introuvable ou expiré."}, 404
    return {"status": job["status"], "progress": job["progress"],
            "image": job["image"], "error": job["error"]}


# ============================================================
# DEBUG + FICHIERS STATIQUES
# ============================================================
@app.route("/debug")
def debug():
    return {
        "key_present": bool(NVIDIA_API_KEY),
        "key_length": len(NVIDIA_API_KEY),
        "key_valid": NVIDIA_API_KEY.startswith("nvapi-") and len(NVIDIA_API_KEY) > 50,
        "active_text_model": ACTIVE_TEXT_MODEL,
        "active_image_model": ACTIVE_IMAGE_MODEL,
        "text_models_count": len(TEXT_MODELS),
        "image_models_count": len(IMAGE_MODELS),
        "pypdf_ok": PYPDF_OK,
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
    print(f"[NOVA] Modèles texte : {len(TEXT_MODELS)}", flush=True)
    print(f"[NOVA] Modèles image : {len(IMAGE_MODELS)}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
