import os
import io
import uuid
import base64
import threading
import time
import re
from datetime import datetime
from urllib.parse import unquote, quote_plus
from flask import Flask, request, Response, stream_with_context, send_from_directory, jsonify
from flask_cors import CORS
import requests

try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_URL = f"{NVIDIA_BASE}/chat/completions"
NVIDIA_IMAGE_BASE = "https://ai.api.nvidia.com/v1/genai"

# ============================================================
# MODES DE RÉPONSE
# ============================================================
MODES = {
    "faible": {"max_tokens": 2000, "temperature": 0.3, "suffix": " Reponds de facon concise."},
    "moyen":  {"max_tokens": 4000, "temperature": 0.7, "suffix": " Sois clair et equilibre."},
    "max":    {"max_tokens": 8000, "temperature": 1.0, "suffix": " Analyse en profondeur."}
}

# ============================================================
# CATALOGUE CURÉ — Modèles principaux uniquement
# ============================================================
# Structure : id -> {name, category, description, best_for}
CURATED_TEXT_MODELS = {
    # -------- RAPIDES --------
    "nvidia/nemotron-3.5-lightning-30b-a3b": {
        "name": "Nemotron 3.5 Lightning",
        "category": "⚡ Rapide",
        "category_key": "rapide",
        "description": "Ultra-rapide, idéal pour les conversations fluides du quotidien.",
        "best_for": "Chat quotidien, réponses instantanées",
    },
    "meta/llama-3.1-8b-instruct": {
        "name": "Llama 3.1 8B",
        "category": "⚡ Rapide",
        "category_key": "rapide",
        "description": "Léger et très rapide, parfait pour les tâches simples.",
        "best_for": "Réponses courtes, questions simples",
    },
    # -------- ÉQUILIBRÉS --------
    "meta/llama-3.3-70b-instruct": {
        "name": "Llama 3.3 70B",
        "category": "⚖️ Équilibré",
        "category_key": "equilibre",
        "description": "Le meilleur compromis vitesse / qualité en 2026.",
        "best_for": "Usage général, code, rédaction",
    },
    "nvidia/llama-3.1-nemotron-70b-instruct": {
        "name": "Nemotron 70B",
        "category": "⚖️ Équilibré",
        "category_key": "equilibre",
        "description": "Grand modèle NVIDIA, très bon en raisonnement.",
        "best_for": "Analyse, rédaction, code",
    },
    # -------- PUISSANTS --------
    "nvidia/nemotron-4-340b-instruct": {
        "name": "Nemotron 4 340B",
        "category": "💪 Puissant",
        "category_key": "puissant",
        "description": "Le plus puissant du catalogue, pour les tâches complexes.",
        "best_for": "Recherche, raisonnement avancé, code complexe",
    },
    # -------- MULTITÂCHE (vision) --------
    "meta/llama-3.2-11b-vision-instruct": {
        "name": "Llama 3.2 Vision 11B",
        "category": "🎨 Multitâche",
        "category_key": "multitache",
        "description": "Comprend à la fois le texte ET les images.",
        "best_for": "Analyse d'images, description visuelle",
    },
    "meta/llama-3.2-90b-vision-instruct": {
        "name": "Llama 3.2 Vision 90B",
        "category": "🎨 Multitâche",
        "category_key": "multitache",
        "description": "Version puissante de Vision, analyse fine des images.",
        "best_for": "Analyse d'images complexe, OCR avancé",
    },
    # -------- RAISONNEMENT --------
    "deepseek-ai/deepseek-r1": {
        "name": "DeepSeek R1",
        "category": "🧠 Raisonnement",
        "category_key": "raisonnement",
        "description": "Excelle en maths, logique et réflexion étape par étape.",
        "best_for": "Maths, problèmes complexes, code algorithmique",
    },
}

# Modèles image principaux
CURATED_IMAGE_MODELS = {
    "black-forest-labs/flux.2-klein-4b": {
        "name": "FLUX.2 Klein 4B",
        "category": "⚡ Rapide",
        "category_key": "rapide",
        "description": "Ultra-rapide (4 étapes), génération en quelques secondes.",
        "best_for": "Génération rapide, style artistique",
    },
    "black-forest-labs/flux.1-schnell": {
        "name": "FLUX.1 Schnell",
        "category": "⚖️ Équilibré",
        "category_key": "equilibre",
        "description": "Très bonne qualité en 4 étapes, style photoréaliste.",
        "best_for": "Photo, qualité rapide",
    },
    "stabilityai/stable-diffusion-3.5-large": {
        "name": "Stable Diffusion 3.5",
        "category": "💪 Puissant",
        "category_key": "puissant",
        "description": "Excellent rendu des détails et du texte dans l'image.",
        "best_for": "Illustrations détaillées, affiches",
    },
}

# Catalogue dynamique (sera filtré)
TEXT_MODELS = {}
IMAGE_MODELS = {}
ACTIVE_TEXT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
ACTIVE_IMAGE_MODEL = "black-forest-labs/flux.2-klein-4b"

FLUX_MAX_PROMPT_LEN = 780
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 120

IMAGE_JOBS = {}


# ============================================================
# CATALOGUE DYNAMIQUE (filtré)
# ============================================================
def _classify_model(model_id):
    mid = model_id.lower()
    image_keywords = ["flux", "diffusion", "sdxl", "dall", "genai", "image"]
    for kw in image_keywords:
        if kw in mid:
            return "image"
    chat_keywords = ["instruct", "chat", "llm", "nemotron", "llama", "qwen", "mistral",
                     "gemma", "deepseek", "phi", "command", "yi", "solar", "zephyr",
                     "vicuna", "falcon", "mpt", "olmo", "r1"]
    for kw in chat_keywords:
        if kw in mid:
            return "chat"
    return "other"


def _build_model_entry(mid, curated_info, kind):
    entry = {
        "id": mid,
        "name": curated_info.get("name") or mid.split("/")[-1].replace("-", " ").title(),
        "provider": mid.split("/")[0].title() if "/" in mid else "NVIDIA",
        "category": curated_info.get("category", ""),
        "category_key": curated_info.get("category_key", ""),
        "description": curated_info.get("description", "Modèle disponible via l'API NVIDIA."),
        "best_for": curated_info.get("best_for", ""),
        "queue": "Variable",
        "speed": "Variable",
        "efficiency": "Variable",
        "default": False,
    }
    if kind == "image":
        entry["endpoint"] = f"{NVIDIA_IMAGE_BASE}/{mid}"
        entry["payload_format"] = "sdxl" if "stabilityai" in mid.lower() else "flux"
        entry["steps"] = 30 if "stabilityai" in mid.lower() else 4
    return entry


def fetch_nvidia_models():
    """Récupère la liste réelle et la filtre au catalogue curé."""
    global TEXT_MODELS, IMAGE_MODELS
    if not NVIDIA_API_KEY:
        return
    try:
        r = requests.get(
            f"{NVIDIA_BASE}/models",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json"},
            timeout=(10, 30)
        )
        if r.status_code != 200:
            print(f"[NOVA] Erreur /v1/models : {r.status_code}", flush=True)
            return
        data = r.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        available_ids = {m.get("id", "") for m in models}
        print(f"[NOVA] {len(available_ids)} modèles disponibles chez NVIDIA.", flush=True)

        # Filtre : ne garde que les modèles curés qui existent réellement
        text_models = {}
        for mid, info in CURATED_TEXT_MODELS.items():
            if mid in available_ids:
                text_models[mid] = _build_model_entry(mid, info, "chat")
            else:
                print(f"[NOVA] ⚠ Texte absent : {mid}", flush=True)

        image_models = {}
        for mid, info in CURATED_IMAGE_MODELS.items():
            if mid in available_ids:
                image_models[mid] = _build_model_entry(mid, info, "image")
            else:
                print(f"[NOVA] ⚠ Image absente : {mid}", flush=True)

        # Fallback : si aucun modèle curé n'est disponible, prend le premier disponible
        if not text_models:
            for m in models:
                mid = m.get("id", "")
                if _classify_model(mid) == "chat":
                    text_models[mid] = _build_model_entry(mid, {"name": mid.split("/")[-1]}, "chat")
                    break

        TEXT_MODELS = text_models
        IMAGE_MODELS = image_models

        if ACTIVE_TEXT_MODEL not in TEXT_MODELS and TEXT_MODELS:
            ACTIVE_TEXT_MODEL = next(iter(TEXT_MODELS))
        if ACTIVE_IMAGE_MODEL not in IMAGE_MODELS and IMAGE_MODELS:
            ACTIVE_IMAGE_MODEL = next(iter(IMAGE_MODELS))

        print(f"[NOVA] Catalogue final : {len(TEXT_MODELS)} texte, {len(IMAGE_MODELS)} image.", flush=True)
    except Exception as e:
        print(f"[NOVA] Exception /v1/models : {e}", flush=True)


fetch_nvidia_models()


# ============================================================
# RECHERCHE WEB (DuckDuckGo, gratuit, sans clé)
# ============================================================
def web_search(query, max_results=3):
    """Recherche web via DuckDuckGo HTML (gratuit). Retourne une liste {title, url, snippet, content}."""
    if not query.strip():
        return []
    try:
        # 1. Recherche
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            },
            timeout=15
        )
        if r.status_code != 200:
            print(f"[NOVA][SEARCH] DDG status {r.status_code}", flush=True)
            return []

        results = []
        if BS4_OK:
            soup = BeautifulSoup(r.text, "html.parser")
            for el in soup.select(".result")[:max_results]:
                a = el.select_one(".result__a")
                snippet_el = el.select_one(".result__snippet")
                if not a:
                    continue
                url = a.get("href", "")
                # DDG enveloppe les URLs : //duckduckgo.com/l/?uddg=<vraie_url>
                if "uddg=" in url:
                    m = re.search(r"uddg=([^&]+)", url)
                    if m:
                        url = unquote(m.group(1))
                results.append({
                    "title": a.get_text(strip=True),
                    "url": url,
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })
        else:
            # Fallback regex basique si BeautifulSoup absent
            for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text)[:max_results]:
                url = m.group(1)
                if "uddg=" in url:
                    m2 = re.search(r"uddg=([^&]+)", url)
                    if m2:
                        url = unquote(m2.group(1))
                results.append({"title": re.sub(r"<[^>]+>", "", m.group(2)), "url": url, "snippet": ""})

        print(f"[NOVA][SEARCH] '{query[:50]}' -> {len(results)} résultats", flush=True)

        # 2. Extraction rapide du contenu (2 premiers résultats)
        for i, res in enumerate(results[:2]):
            try:
                rr = requests.get(
                    res["url"],
                    headers={"User-Agent": "Mozilla/5.0 (compatible; NOVA/1.0)"},
                    timeout=8
                )
                if rr.status_code == 200:
                    if BS4_OK:
                        s = BeautifulSoup(rr.text, "html.parser")
                        for tag in s(["script", "style", "nav", "footer", "header", "aside"]):
                            tag.decompose()
                        text = " ".join(s.get_text(separator=" ").split())
                    else:
                        text = re.sub(r"<[^>]+>", " ", rr.text)
                        text = " ".join(text.split())
                    res["content"] = text[:1500]
                else:
                    res["content"] = res.get("snippet", "")
            except Exception as e:
                res["content"] = res.get("snippet", "")

        return results
    except Exception as e:
        print(f"[NOVA][SEARCH] Exception : {e}", flush=True)
        return []


# ============================================================
# ROUTES CATALOGUE
# ============================================================
@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({
        "text_models": list(TEXT_MODELS.values()),
        "image_models": list(IMAGE_MODELS.values()),
        "active_text_model": ACTIVE_TEXT_MODEL,
        "active_image_model": ACTIVE_IMAGE_MODEL,
    })


@app.route("/api/models/select", methods=["POST"])
def select_model():
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
# UTILITAIRES
# ============================================================
def _truncate_prompt(prompt, max_len=FLUX_MAX_PROMPT_LEN):
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
            "en français. Décris UNIQUEMENT : formes, couleurs, style, ambiance, décor. "
            f"Maximum {max_words} mots. Ne mentionne NI personne, NI visage, NI marque. "
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
            "max_tokens": 500,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        r = requests.post(NVIDIA_CHAT_URL, json=payload, headers=headers, timeout=(15, 90))
        if r.status_code != 200:
            print(f"[NOVA][VISION ERR] {r.status_code}", flush=True)
            return ""
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return (content or "").strip()
    except Exception as e:
        print(f"[NOVA][VISION EXCEPTION] {e}", flush=True)
        return ""


# ============================================================
# ROUTE CHAT (avec recherche web optionnelle)
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
    use_web_search = bool(data.get("web_search", False))

    cfg = MODES.get(mode, MODES["moyen"])
    ai_name = config.get("aiName", "NOVA")
    personality = config.get("personality", "Tu es un assistant IA utile et bienveillant.")
    temperature = float(config.get("temperature", cfg["temperature"]))
    max_tokens = min(int(config.get("maxTokens", cfg["max_tokens"])), cfg["max_tokens"])

    now = datetime.now()
    date_str = f"{now.day}/{now.month}/{now.year}"

    # === Recherche web si demandée ===
    search_context = ""
    last_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_text = m.get("content", "")
            break

    if use_web_search and last_user_text.strip():
        results = web_search(last_user_text, max_results=3)
        if results:
            parts = []
            for i, r in enumerate(results, 1):
                parts.append(f"[Source {i}] {r['title']}\nURL : {r['url']}\n{r.get('content') or r.get('snippet','')}")
            search_context = (
                "Voici des informations récentes trouvées sur Internet "
                f"(nous sommes le {date_str}). Base ta réponse sur ces informations "
                "quand c'est pertinent, et cite tes sources en mentionnant l'URL :\n\n"
                + "\n\n---\n\n".join(parts)
            )
            print(f"[NOVA][CHAT] Recherche web activée : {len(results)} sources", flush=True)

    system_prompt = (
        f"Tu t'appelles {ai_name}. {personality} "
        f"Tu reponds TOUJOURS en francais. "
        f"{cfg['suffix']} Nous sommes le {date_str}."
    )

    has_image = any(a.get("type") == "image" for a in attachments)
    if has_image and ACTIVE_TEXT_MODEL != "meta/llama-3.2-11b-vision-instruct":
        model = "meta/llama-3.2-11b-vision-instruct"
    else:
        model = ACTIVE_TEXT_MODEL

    trimmed = messages[-20:] if len(messages) > 20 else messages[:]
    final_messages = [{"role": "system", "content": system_prompt}]

    # Injecte le contexte web comme message système additionnel
    if search_context:
        final_messages.append({"role": "system", "content": search_context})

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
                        "text": f"\n\n--- {att.get('filename','document')} ---\n{att.get('content','')}\n---\n"
                    })
                elif att.get("type") == "image":
                    desc = att.get("description")
                    if desc:
                        content_parts.append({"type": "text", "text": f"\n[Description image : {desc}]\n"})
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
# ROUTE RECHERCHE WEB DIRECTE (pour test)
# ============================================================
@app.route("/api/search", methods=["POST"])
def search_endpoint():
    data = request.get_json() or {}
    q = (data.get("query") or "").strip()
    if not q:
        return {"error": "Requête vide."}, 400
    results = web_search(q, max_results=5)
    return {"query": q, "results": results}


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
            return {"error": "Support PDF indisponible."}, 500
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for page in reader.pages[:50]:
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception:
                    pages_text.append("")
            text = "\n".join(pages_text).strip()
            if not text:
                return {"error": "Aucun texte extractible."}, 400
            return {"type": "text", "filename": filename, "size": size,
                    "content": text[:50000], "truncated": len(text) > 50000}
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
        return {"type": "text", "filename": filename, "size": size,
                "content": text[:50000], "truncated": len(text) > 50000}

    if ext in IMAGE_EXTENSIONS:
        if size > 8 * 1024 * 1024:
            return {"error": "Image trop volumineuse (max 8 Mo)."}, 400
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}.get(ext, "image/png")
        b64 = base64.b64encode(file_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        description = _describe_image(data_url)
        return {"type": "image", "filename": filename, "size": size,
                "mime": mime, "data_url": data_url, "description": description}

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
            IMAGE_JOBS[job_id]["error"] = "🚫 Description bloquée par le filtre NVIDIA."
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
        IMAGE_JOBS[job_id]["status"] = "error"
        IMAGE_JOBS[job_id]["error"] = "Format de réponse inconnu."
        return
    IMAGE_JOBS[job_id]["status"] = "done"
    IMAGE_JOBS[job_id]["progress"] = 100
    IMAGE_JOBS[job_id]["image"] = image_url


def _run_image_job(job_id, user_prompt):
    try:
        final_prompt = user_prompt.strip() or "a beautiful abstract art piece"
        final_prompt = _truncate_prompt(final_prompt, max_len=FLUX_MAX_PROMPT_LEN)
        IMAGE_JOBS[job_id]["status"] = "generating"
        IMAGE_JOBS[job_id]["progress"] = 30

        model_info = IMAGE_MODELS.get(ACTIVE_IMAGE_MODEL)
        if not model_info:
            IMAGE_JOBS[job_id]["status"] = "error"
            IMAGE_JOBS[job_id]["error"] = "Aucun modèle image actif."
            return

        endpoint = model_info["endpoint"]
        fmt = model_info.get("payload_format", "flux")

        if fmt == "sdxl":
            payload = {
                "text_prompts": [{"text": final_prompt}],
                "cfg_scale": 5.0, "seed": 0, "steps": 30,
                "width": 1024, "height": 1024,
            }
        else:
            payload = {
                "prompt": final_prompt,
                "width": 1024, "height": 1024, "steps": 4, "seed": 0,
            }

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        r = requests.post(endpoint, json=payload, headers=headers, timeout=(20, 300))
        IMAGE_JOBS[job_id]["progress"] = 85
        if r.status_code != 200:
            err = r.text[:300]
            IMAGE_JOBS[job_id]["status"] = "error"
            IMAGE_JOBS[job_id]["error"] = f"API {r.status_code}: {err}"
            return
        _save_image_result(job_id, r)
    except Exception as e:
        IMAGE_JOBS[job_id]["status"] = "error"
        IMAGE_JOBS[job_id]["error"] = str(e)[:200]


@app.route("/api/image/start", methods=["POST"])
def image_start():
    if not NVIDIA_API_KEY:
        return {"error": "Cle API manquante."}, 500
    data = request.get_json() or {}
    user_prompt = (data.get("prompt") or "").strip()
    ref_desc = (data.get("reference_description") or "").strip()
    if not user_prompt and not ref_desc:
        return {"error": "Prompt vide."}, 400
    if not user_prompt:
        user_prompt = ref_desc[:200]

    job_id = str(uuid.uuid4())
    IMAGE_JOBS[job_id] = {"status": "pending", "progress": 0, "image": None,
                          "error": None, "created": time.time()}
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
        return {"error": "Job introuvable."}, 404
    return {"status": job["status"], "progress": job["progress"],
            "image": job["image"], "error": job["error"]}


# ============================================================
# DEBUG + FICHIERS STATIQUES
# ============================================================
@app.route("/debug")
def debug():
    return {
        "key_present": bool(NVIDIA_API_KEY),
        "key_valid": NVIDIA_API_KEY.startswith("nvapi-"),
        "text_models_count": len(TEXT_MODELS),
        "image_models_count": len(IMAGE_MODELS),
        "text_models_ids": list(TEXT_MODELS.keys()),
        "image_models_ids": list(IMAGE_MODELS.keys()),
        "active_text_model": ACTIVE_TEXT_MODEL,
        "active_image_model": ACTIVE_IMAGE_MODEL,
        "pypdf_ok": PYPDF_OK,
        "bs4_ok": BS4_OK,
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
