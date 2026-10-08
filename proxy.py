import os
import io
import uuid
import base64
import threading
import time
from datetime import datetime
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import requests

try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 Mo max

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()

# ====== CHAT ======
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_CHAT = os.environ.get("NVIDIA_MODEL_CHAT", "nvidia/nemotron-3.5-lightning-30b-a3b")

# ====== VISION (analyse des images) ======
MODEL_VISION = os.environ.get("NVIDIA_MODEL_VISION", "meta/llama-3.2-11b-vision-instruct")

# ====== IMAGE (génération FLUX.2-klein-4B) ======
NVIDIA_IMAGE_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b"

MODES = {
    "faible": {"max_tokens": 2000, "temperature": 0.3, "suffix": " Reponds de facon concise."},
    "moyen":  {"max_tokens": 4000, "temperature": 0.7, "suffix": " Sois clair et equilibre."},
    "max":    {"max_tokens": 8000, "temperature": 1.0, "suffix": " Analyse en profondeur."}
}

CONNECT_TIMEOUT = 15
READ_TIMEOUT = 120

IMAGE_JOBS = {}

# ============================================================
# HELPERS — DESCRIPTION D'IMAGE (VISION)
# ============================================================
def _describe_image(data_url, max_words=70):
    """
    Envoie l'image au modele vision et retourne une description textuelle.
    data_url : "data:image/png;base64,..."
    Retourne "" si echec.
    """
    if not NVIDIA_API_KEY or not data_url:
        return ""
    try:
        prompt = (
            "Tu es un assistant qui décrit précisément les images. "
            f"Décris cette image en français en {max_words} mots maximum. "
            "Concentre-toi sur : le sujet principal, les couleurs dominantes, "
            "le style (photo, dessin, 3D...), l'ambiance et la composition. "
            "Réponds uniquement par la description, sans introduction."
        )
        payload = {
            "model": MODEL_VISION,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            }],
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 200,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        r = requests.post(NVIDIA_CHAT_URL, json=payload, headers=headers,
                          timeout=(10, 60))
        if r.status_code != 200:
            print(f"[NOVA][VISION ERR] {r.status_code} -> {r.text[:200]}", flush=True)
            return ""
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return (content or "").strip()
    except Exception as e:
        print(f"[NOVA][VISION EXCEPTION] {e}", flush=True)
        return ""


# ============================================================
# UPLOAD DE FICHIERS
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

    # --- PDF ---
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
                "type": "text",
                "filename": filename,
                "size": size,
                "content": text[:50000],
                "truncated": len(text) > 50000,
            }
        except Exception as e:
            return {"error": f"Erreur PDF : {e}"}, 500

    # --- Texte ---
    if ext in TEXT_EXTENSIONS or (ext == "" and size < 100_000):
        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            try:
                text = file_bytes.decode("latin-1", errors="replace")
            except Exception as e:
                return {"error": f"Décodage impossible : {e}"}, 500
        return {
            "type": "text",
            "filename": filename,
            "size": size,
            "content": text[:50000],
            "truncated": len(text) > 50000,
        }

    # --- Image ---
    if ext in IMAGE_EXTENSIONS:
        if size > 8 * 1024 * 1024:
            return {"error": "Image trop volumineuse (max 8 Mo)."}, 400
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
        }.get(ext, "image/png")
        b64 = base64.b64encode(file_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        # 🔑 CONVERSION AUTOMATIQUE EN TEXTE (invisible pour l'utilisateur)
        description = _describe_image(data_url)
        print(f"[NOVA][UPLOAD IMAGE] {filename} | desc = {description[:100]}...", flush=True)

        return {
            "type": "image",
            "filename": filename,
            "size": size,
            "mime": mime,
            "data_url": data_url,
            "description": description,   # <- description cachée
        }

    return {"error": f"Type de fichier non supporté : {ext or '?'}"}, 400


# ============================================================
# ROUTE CHAT (avec support des fichiers joints + descriptions)
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
    model = MODEL_VISION if has_image else MODEL_CHAT

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
                    # Texte "caché" : description générée à l'upload
                    desc = att.get("description") or att.get("_description")
                    if desc:
                        content_parts.append({
                            "type": "text",
                            "text": (
                                f"\n\n[Description automatique de l'image "
                                f"« {att.get('filename','image')} » : {desc}]\n"
                            )
                        })
                    # L'image elle-même pour la vision
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
                               stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
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
# GÉNÉRATION D'IMAGES — Job + Polling
# ============================================================

def _save_image_result(job_id, r):
    """Extrait l'image de la reponse NVIDIA et met a jour le job."""
    result = r.json()
    image_url = None

    artifacts = result.get("artifacts") or []
    if artifacts and isinstance(artifacts[0], dict):
        art = artifacts[0]
        reason = art.get("finishReason") or art.get("finish_reason")
        if reason == "CONTENT_FILTERED":
            IMAGE_JOBS[job_id]["status"] = "error"
            IMAGE_JOBS[job_id]["error"] = (
                "🚫 Description bloquée par le filtre NVIDIA. Reformule sans éléments sensibles."
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


def _run_image_job(job_id, prompt):
    try:
        IMAGE_JOBS[job_id]["status"] = "generating"
        IMAGE_JOBS[job_id]["progress"] = 10

        payload = {
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 0
        }
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        IMAGE_JOBS[job_id]["progress"] = 30
        r = requests.post(NVIDIA_IMAGE_URL, json=payload, headers=headers,
                          timeout=(20, 300))
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
    prompt = (data.get("prompt") or "").strip()
    # 🔑 Description d'image eventuelle (generee a l'upload)
    reference_description = (data.get("reference_description") or "").strip()

    # 🔑 On combine : prompt utilisateur + description cachee de l'image
    if reference_description:
        combined = (
            f"{prompt}\n\n"
            f"[Style et contenu inspirés de l'image de référence : {reference_description}]"
            if prompt else
            f"Crée une image inspirée de cette description : {reference_description}"
        )
    else:
        combined = prompt

    if not combined.strip():
        return {"error": "Prompt vide."}, 400

    job_id = str(uuid.uuid4())
    IMAGE_JOBS[job_id] = {
        "status": "pending", "progress": 0,
        "image": None, "error": None, "created": time.time(),
    }
    threading.Thread(target=_run_image_job, args=(job_id, combined), daemon=True).start()

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
        "chat_model": MODEL_CHAT,
        "vision_model": MODEL_VISION,
        "image_url": NVIDIA_IMAGE_URL,
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
    print(f"[NOVA] Chat   : {MODEL_CHAT}", flush=True)
    print(f"[NOVA] Vision : {MODEL_VISION}", flush=True)
    print(f"[NOVA] Image  : {NVIDIA_IMAGE_URL}", flush=True)
    print(f"[NOVA] pypdf  : {PYPDF_OK}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
