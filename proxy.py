"""
proxy_debug.py — Script de diagnostic pour l'API image NVIDIA
Usage : python proxy_debug.py
Puis ouvre http://localhost:5001 (ou l'URL de ton déploiement)
"""
import os
import time
import json
import base64
from flask import Flask, request, jsonify, Response

import requests

app = Flask(__name__)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()

# URLs candidates pour les modèles image NVIDIA NIM
IMAGE_MODELS = {
    "flux_schnell": "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
    "flux_dev":     "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev",
    "sdxl":         "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl",
    "sdxl_turbo":   "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo",
}


# ============================================================
# PAGE D'ACCUEIL — Interface de diagnostic
# ============================================================
@app.route("/")
def index():
    html = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>🔧 Debug API Image NVIDIA</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 960px;
         margin: 30px auto; padding: 20px; background:#faf9f5; color:#1a1915; line-height:1.6; }
  h1 { font-size: 22px; margin-bottom: 6px; }
  h2 { font-size: 16px; margin-top: 28px; padding-bottom: 6px;
       border-bottom: 2px solid #e8e3d5; color:#C96442; }
  .card { background:#fff; border:1px solid #e8e3d5; border-radius:12px;
          padding:18px; margin:12px 0; }
  .env-ok { color:#16a34a; font-weight:600; }
  .env-ko { color:#dc2626; font-weight:600; }
  pre { background:#f5f2eb; border:1px solid #e8e3d5; padding:12px;
        border-radius:8px; overflow-x:auto; font-size:12.5px;
        font-family: 'SF Mono', Menlo, monospace; max-height:400px; }
  input, select, button { padding:9px 12px; font-size:14px; border-radius:8px;
                          border:1px solid #e8e3d5; font-family:inherit; }
  input[type=text] { width: 60%; }
  button { background:#C96442; color:#fff; border:none; cursor:pointer;
           font-weight:600; padding:9px 18px; }
  button:hover { background:#b35638; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:10px 0; }
  .status { padding:6px 12px; border-radius:6px; font-size:13px; font-weight:600; }
  .status.ok { background:#dcfce7; color:#166534; }
  .status.err { background:#fee2e2; color:#991b1b; }
  .status.wait { background:#fef3c7; color:#92400e; }
  .kv { display:grid; grid-template-columns: 180px 1fr; gap:6px 14px;
        font-size:13.5px; margin:10px 0; }
  .kv b { color:#5c5a52; font-weight:600; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  img.preview { max-width: 100%; border-radius:10px; border:1px solid #e8e3d5;
                margin-top:12px; background:#fff; }
  @media(max-width:700px){ .grid{grid-template-columns:1fr;} input[type=text]{width:100%;} }
</style>
</head>
<body>

<h1>🔧 Debug — API Image NVIDIA</h1>
<p style="color:#5c5a52;">Page de diagnostic. Elle teste l'API image NVIDIA et affiche la réponse brute.</p>

<div class="card">
  <h2 style="margin-top:0;border:none;">1. État de la configuration</h2>
  <div class="kv">
    <b>Clé API présente</b>  <span id="env-key">…</span>
    <b>Longueur clé</b>       <span id="env-len">…</span>
    <b>Clé valide</b>         <span id="env-valid">…</span>
    <b>Python</b>             <span id="env-py">…</span>
    <b>Version requests</b>   <span id="env-req">…</span>
  </div>
  <button onclick="loadEnv()">Rafraîchir</button>
</div>

<div class="card">
  <h2 style="margin-top:0;border:none;">2. Test d'un modèle</h2>
  <div class="row">
    <select id="model-select">
      <option value="">-- choisis un modèle --</option>
    </select>
    <input type="text" id="prompt-input" value="a cute cat astronaut on the moon" placeholder="Prompt anglais conseillé">
    <button id="test-btn" onclick="runTest()">Tester</button>
  </div>
  <div id="test-status" class="status wait">En attente…</div>
  <div id="test-result" style="margin-top:14px;"></div>
</div>

<div class="card">
  <h2 style="margin-top:0;border:none;">3. Test manuel d'une URL</h2>
  <div class="row">
    <input type="text" id="custom-url" placeholder="https://ai.api.nvidia.com/v1/genai/..." style="width:70%;">
    <button onclick="runCustom()">Tester cette URL</button>
  </div>
  <div id="custom-result" style="margin-top:14px;"></div>
</div>

<div class="card">
  <h2 style="margin-top:0;border:none;">4. Liste des modèles disponibles</h2>
  <button onclick="listModels()">Interroger /v1/models</button>
  <pre id="models-list" style="margin-top:12px;">Clique sur le bouton pour voir la liste.</pre>
</div>

<script>
async function loadEnv(){
  const r = await fetch('/debug/env');
  const j = await r.json();
  const ok = '<span class="env-ok">✅</span>';
  const ko = '<span class="env-ko">❌</span>';
  document.getElementById('env-key').innerHTML = j.key_present ? ok+' présente' : ko+' manquante';
  document.getElementById('env-len').textContent = j.key_length + ' chars';
  document.getElementById('env-valid').innerHTML = j.key_valid ? ok : ko;
  document.getElementById('env-py').textContent = j.python_version;
  document.getElementById('env-req').textContent = j.requests_version;

  const sel = document.getElementById('model-select');
  sel.innerHTML = '<option value="">-- choisis un modèle --</option>';
  for (const [name, url] of Object.entries(j.image_models)){
    const o = document.createElement('option');
    o.value = url; o.textContent = name + ' — ' + url;
    sel.appendChild(o);
  }
}

async function runTest(){
  const url = document.getElementById('model-select').value;
  const prompt = document.getElementById('prompt-input').value;
  if (!url) { alert('Choisis un modèle'); return; }
  await doTest(url, prompt, 'test-status', 'test-result');
}

async function runCustom(){
  const url = document.getElementById('custom-url').value.trim();
  const prompt = document.getElementById('prompt-input').value;
  if (!url) { alert('Colle une URL'); return; }
  await doTest(url, prompt, null, 'custom-result');
}

async function doTest(url, prompt, statusId, resultId){
  const resultEl = document.getElementById(resultId);
  const statusEl = statusId ? document.getElementById(statusId) : null;
  if (statusEl) { statusEl.className = 'status wait'; statusEl.textContent = 'En cours…'; }
  resultEl.innerHTML = '<pre>Requête envoyée, attente…</pre>';

  const btn = document.getElementById('test-btn');
  if (btn) btn.disabled = true;

  const t0 = Date.now();
  try {
    const r = await fetch('/debug/image', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url, prompt})
    });
    const elapsed = ((Date.now() - t0)/1000).toFixed(2);
    const j = await r.json();

    if (statusEl) {
      if (j.ok) { statusEl.className = 'status ok'; statusEl.textContent = '✅ OK — ' + elapsed + 's'; }
      else      { statusEl.className = 'status err'; statusEl.textContent = '❌ Échec — ' + elapsed + 's'; }
    }

    let html = '<div class="kv">';
    html += '<b>URL testée</b><span style="word-break:break-all;">' + escapeHtml(url) + '</span>';
    html += '<b>Statut HTTP</b><span>' + (j.status_code || '—') + '</span>';
    html += '<b>Durée</b><span>' + elapsed + ' s</span>';
    html += '<b>Content-Type</b><span>' + (j.content_type || '—') + '</span>';
    if (j.error) html += '<b>Erreur</b><span style="color:#dc2626;">' + escapeHtml(j.error) + '</span>';
    if (j.image_found !== undefined)
      html += '<b>Image trouvée</b><span>' + (j.image_found ? '✅ oui' : '❌ non') + '</span>';
    if (j.image_format)
      html += '<b>Format image</b><span>' + escapeHtml(j.image_format) + '</span>';
    html += '</div>';

    if (j.image_preview) {
      html += '<div style="margin-top:10px;"><b>Aperçu :</b><br>';
      html += '<img class="preview" src="' + j.image_preview + '"></div>';
    }

    if (j.raw_response !== undefined) {
      html += '<h4 style="margin-top:16px;">Réponse brute (500 premiers caractères)</h4>';
      html += '<pre>' + escapeHtml(typeof j.raw_response === 'string' ? j.raw_response : JSON.stringify(j.raw_response, null, 2)) + '</pre>';
    }
    resultEl.innerHTML = html;

  } catch(e){
    if (statusEl) { statusEl.className = 'status err'; statusEl.textContent = '❌ Erreur : ' + e.message; }
    resultEl.innerHTML = '<pre>' + escapeHtml(String(e)) + '</pre>';
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function listModels(){
  const el = document.getElementById('models-list');
  el.textContent = 'Chargement…';
  try {
    const r = await fetch('/debug/models');
    const j = await r.json();
    el.textContent = JSON.stringify(j, null, 2);
  } catch(e){
    el.textContent = 'Erreur : ' + e.message;
  }
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

window.addEventListener('load', loadEnv);
</script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


# ============================================================
# ENV
# ============================================================
@app.route("/debug/env")
def debug_env():
    import sys
    try:
        import requests as _r
        req_ver = _r.__version__
    except Exception:
        req_ver = "?"
    return jsonify({
        "key_present": bool(NVIDIA_API_KEY),
        "key_length": len(NVIDIA_API_KEY),
        "key_valid": NVIDIA_API_KEY.startswith("nvapi-") and len(NVIDIA_API_KEY) > 50,
        "python_version": sys.version.split()[0],
        "requests_version": req_ver,
        "image_models": IMAGE_MODELS,
    })


# ============================================================
# TEST IMAGE
# ============================================================
@app.route("/debug/image", methods=["POST"])
def debug_image():
    data = request.get_json() or {}
    url = data.get("url") or IMAGE_MODELS["flux_schnell"]
    prompt = (data.get("prompt") or "a cat").strip()

    if not NVIDIA_API_KEY:
        return jsonify({"ok": False, "error": "Clé API manquante (NVIDIA_API_KEY)."})

    payload = {
        "prompt": prompt,
        "mode": "base",
        "seed": 0,
        "steps": 4,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=(20, 150))
        elapsed = time.time() - t0

        # Tentative de parsing
        image_preview = None
        image_format = None
        image_found = False
        parsed = None
        try:
            parsed = r.json()
        except Exception:
            parsed = None

        if parsed and r.status_code == 200:
            artifacts = parsed.get("artifacts") or []
            if artifacts and isinstance(artifacts[0], dict):
                art = artifacts[0]
                if art.get("base64"):
                    image_preview = "data:image/png;base64," + art["base64"]
                    image_format = "artifacts[0].base64"
                    image_found = True
                elif art.get("url"):
                    image_preview = art["url"]
                    image_format = "artifacts[0].url"
                    image_found = True
            if not image_found and parsed.get("image"):
                img = parsed["image"]
                image_preview = img if img.startswith("data:") else "data:image/png;base64," + img
                image_format = "root.image"
                image_found = True

        return jsonify({
            "ok": r.status_code == 200 and image_found,
            "status_code": r.status_code,
            "content_type": r.headers.get("Content-Type", ""),
            "elapsed_sec": round(elapsed, 2),
            "image_found": image_found,
            "image_format": image_format,
            "image_preview": image_preview,
            "raw_response": (r.text[:500] if not image_found else
                             json.dumps({k: ("<base64:...>" if k == "image" else v)
                                          for k, v in (parsed or {}).items()}, indent=2)[:500]),
        })

    except requests.exceptions.ConnectTimeout:
        return jsonify({"ok": False, "error": "Timeout de connexion (20s)."})
    except requests.exceptions.ReadTimeout:
        return jsonify({"ok": False, "error": "Timeout de lecture (150s)."})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})


# ============================================================
# LISTE DES MODELES
# ============================================================
@app.route("/debug/models")
def debug_models():
    if not NVIDIA_API_KEY:
        return jsonify({"error": "Clé API manquante."})
    try:
        r = requests.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json"},
            timeout=(15, 30),
        )
        try:
            data = r.json()
        except Exception:
            data = r.text[:1000]
        return jsonify({"status_code": r.status_code, "data": data})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})


# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 60, flush=True)
    print(f"[DEBUG] Clé API : {len(NVIDIA_API_KEY)} chars", flush=True)
    print(f"[DEBUG] Serveur : http://0.0.0.0:{port}", flush=True)
    print("=" * 60, flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
