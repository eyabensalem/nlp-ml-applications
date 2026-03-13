import streamlit as st
import os
from mistralai import Mistral
from dotenv import load_dotenv

st.set_page_config(
    page_title="Mistral · Assistant IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #0d0f14 !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 4rem 5rem !important; max-width: 1200px !important; }

/* ── Sidebar ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0a0c10 !important;
    border-right: 1px solid #1a2030 !important;
}
[data-testid="stSidebar"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #4a5a70 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #111420 !important;
    border: 1px solid #1e2840 !important;
    border-radius: 5px !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
.sidebar-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #3a7a9a;
    display: block;
    margin-bottom: 0.3rem;
}
.sidebar-title {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 300;
    color: #e8e4dc;
    margin: 0 0 1.8rem 0;
}
.task-info {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #3a4a5e;
    line-height: 1.65;
    margin-top: 1.2rem;
    padding: 0.8rem;
    background: #0d1018;
    border: 1px solid #1a2030;
    border-radius: 5px;
}

/* ── Hero ─────────────────────────────────────── */
.hero {
    margin-bottom: 2.8rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #1a2030;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3a7a9a;
    display: block;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 2.8rem;
    font-weight: 300;
    line-height: 1.1;
    color: #f0ece3;
    margin: 0 0 0.6rem 0;
}
.hero-title em { font-style: italic; color: #5aaaca; }
.hero-sub {
    font-size: 0.88rem;
    color: #6b7280;
    font-weight: 300;
    max-width: 580px;
    line-height: 1.65;
}
.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    color: #3a7a9a;
    background: #0d1520;
    border: 1px solid #1e3040;
    padding: 0.28rem 0.75rem;
    border-radius: 20px;
    margin-top: 0.9rem;
}
.model-chip::before { content: '◆'; font-size: 0.5rem; color: #2a5a78; }

/* ── Divider ─────────────────────────────────── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, #1a2030 0%, #1e4060 50%, #1a2030 100%);
    margin: 2rem 0;
    border: none;
}

/* ── Task header ─────────────────────────────── */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 1.1rem;
    margin-bottom: 0.8rem;
}
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #3a7a9a;
    letter-spacing: 0.1em;
    background: #0d1520;
    border: 1px solid #1e3040;
    padding: 0.2rem 0.55rem;
    border-radius: 3px;
    flex-shrink: 0;
}
.section-title {
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 300;
    color: #e8e4dc;
    margin: 0;
}
.section-desc {
    font-size: 0.84rem;
    color: #6b7280;
    line-height: 1.6;
    padding-left: 1rem;
    border-left: 2px solid #1e3040;
    margin-bottom: 1.2rem;
}

/* ── Text inputs ─────────────────────────────── */
.stTextArea > div > div > textarea,
.stTextInput > div > div > input {
    background: #111420 !important;
    border: 1px solid #1a2030 !important;
    border-radius: 6px !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.75 !important;
    padding: 0.9rem 1.1rem !important;
    caret-color: #3a7a9a;
}
.stTextArea > div > div > textarea:focus,
.stTextInput > div > div > input:focus {
    border-color: #1e4060 !important;
    box-shadow: 0 0 0 3px rgba(58,122,154,0.08) !important;
    outline: none !important;
}
.stTextArea label, .stTextInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #3a4a5e !important;
}

/* ── Button ──────────────────────────────────── */
.stButton > button {
    background: #0d1520 !important;
    border: 1px solid #1e4060 !important;
    color: #5aaaca !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 1.8rem !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #0d1a28 !important;
    border-color: #3a7a9a !important;
    color: #8acce8 !important;
    box-shadow: 0 0 20px rgba(58,122,154,0.18) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ── Result area ─────────────────────────────── */
.result-wrap {
    background: #0d1015;
    border: 1px solid #1a2030;
    border-left: 3px solid #2a6080;
    border-radius: 0 6px 6px 0;
    padding: 1.3rem 1.5rem;
    margin-top: 1.4rem;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3a4a5e;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.result-label::after { content: ''; flex:1; height:1px; background:#1a2030; }
.result-text {
    font-size: 0.9rem;
    color: #b8c8d8;
    line-height: 1.8;
    white-space: pre-wrap;
}

/* ── Code block ──────────────────────────────── */
.stCodeBlock {
    border: 1px solid #1a2030 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}
pre {
    background: #0a0e14 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── History ─────────────────────────────────── */
.history-entry {
    background: #0d1015;
    border: 1px solid #1a2030;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.history-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #2a4a62;
    margin-bottom: 0.4rem;
}
.history-q {
    font-size: 0.82rem;
    color: #5a7a8a;
    margin-bottom: 0.3rem;
    font-style: italic;
}
.history-a {
    font-size: 0.84rem;
    color: #8a9aaa;
    line-height: 1.6;
}

/* ── Alerts ──────────────────────────────────── */
.stAlert { background: #0d1015 !important; border: 1px solid #1a2030 !important; border-radius: 6px !important; }
.stAlert p { color: #6b7280 !important; font-size: 0.82rem !important; }
div[data-testid="stWarning"] { background: #14110a !important; border-color: #3a2e10 !important; }

/* ── Expander ────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0d1015 !important;
    border: 1px solid #1a2030 !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #3a5068 !important;
    letter-spacing: 0.06em !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Init
# =========================
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("Clé API Mistral manquante. Ajoutez `MISTRAL_API_KEY` dans votre fichier `.env`.")
    st.stop()

MODEL_NAME = "ministral-8b-2410"

@st.cache_resource(show_spinner=False)
def get_client():
    return Mistral(api_key=api_key)

client = get_client()

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# Sidebar
# =========================
st.sidebar.markdown("""
<span class="sidebar-eyebrow">Configuration</span>
<h2 class="sidebar-title">Choisir une tâche</h2>
""", unsafe_allow_html=True)

task = st.sidebar.selectbox(
    "Tâche",
    ["Chat / Question-Réponse", "Résumé de texte", "Génération de code"],
    label_visibility="collapsed"
)

task_descriptions = {
    "Chat / Question-Réponse": "Pose une question libre. Le modèle répond en français de façon claire et pédagogique.",
    "Résumé de texte": "Colle un texte long. Le modèle en produit un résumé fidèle et concis.",
    "Génération de code": "Décris une fonction ou un programme. Le modèle génère du Python lisible et commenté.",
}
st.sidebar.markdown(f'<div class="task-info">{task_descriptions[task]}</div>', unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("↺ Effacer l'historique", use_container_width=True):
    st.session_state.history = []
    st.rerun()

# =========================
# Hero
# =========================
task_icons = {
    "Chat / Question-Réponse": "dialogue",
    "Résumé de texte": "résumé",
    "Génération de code": "code",
}
st.markdown(f"""
<div class="hero">
    <span class="hero-eyebrow">Natural Language Processing · LLM</span>
    <h1 class="hero-title">Mistral &amp; <em>{task_icons[task]}</em></h1>
    <p class="hero-sub">Interface de génération de texte via l'API Mistral AI. Trois modes : question-réponse, résumé automatique, et génération de code Python.</p>
    <span class="model-chip">{MODEL_NAME}</span>
</div>
""", unsafe_allow_html=True)

# =========================
# Input zone
# =========================
task_tag = {"Chat / Question-Réponse": "01", "Résumé de texte": "02", "Génération de code": "03"}[task]

st.markdown(f"""
<div class="section-header">
    <span class="section-tag">{task_tag}</span>
    <h2 class="section-title">{task}</h2>
</div>
""", unsafe_allow_html=True)

prompt = ""
system_prompt = ""

if task == "Chat / Question-Réponse":
    st.markdown('<p class="section-desc">Posez n\'importe quelle question — le modèle répondra en français.</p>', unsafe_allow_html=True)
    user_input = st.text_input("Votre question", placeholder="Ex : Qu'est-ce que l'attention dans les Transformers ?", label_visibility="visible")
    if user_input:
        system_prompt = "Tu es un assistant utile et pédagogique. Réponds toujours en français, clairement et simplement."
        prompt = f"Réponds à la question suivante :\n\n{user_input}"

elif task == "Résumé de texte":
    st.markdown('<p class="section-desc">Collez un texte brut — le modèle en extraira l\'essentiel.</p>', unsafe_allow_html=True)
    text = st.text_area("Texte à résumer", placeholder="Collez votre texte ici…", height=220, label_visibility="visible")
    if text:
        system_prompt = "Tu es un assistant spécialisé dans le résumé de texte. Réponds en français avec un résumé clair, fidèle et concis."
        prompt = f"Résume le texte suivant en français :\n\n{text}"

elif task == "Génération de code":
    st.markdown('<p class="section-desc">Décrivez en langage naturel ce que vous souhaitez coder.</p>', unsafe_allow_html=True)
    code_request = st.text_area("Description", placeholder="Ex : Une fonction qui nettoie un dataframe pandas en supprimant les doublons et les NaN…", height=180, label_visibility="visible")
    if code_request:
        system_prompt = "Tu es un assistant expert en Python. Génère un code correct, lisible, commenté, et directement exécutable. Réponds uniquement avec le code quand c'est pertinent."
        prompt = f"Génère un code Python pour la demande suivante :\n\n{code_request}"

# =========================
# Generate
# =========================
def call_mistral(prompt: str, system_prompt: str) -> str:
    try:
        resp = client.chat.complete(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Erreur API Mistral : {e}"

col_btn, _ = st.columns([1, 3])
with col_btn:
    generate_clicked = st.button("→ Générer", key="btn_generate")

if generate_clicked:
    if not prompt:
        st.warning("⚠ Remplissez le champ de saisie avant de générer.")
    else:
        with st.spinner("Mistral génère la réponse…"):
            result = call_mistral(prompt, system_prompt)

        st.session_state.history.insert(0, {
            "task": task,
            "prompt": prompt,
            "result": result,
        })

        if task == "Génération de code":
            st.markdown('<div class="result-wrap"><div class="result-label">Code généré</div></div>', unsafe_allow_html=True)
            st.code(result, language="python")
        else:
            st.markdown(f"""
            <div class="result-wrap">
                <div class="result-label">Réponse du modèle</div>
                <div class="result-text">{result}</div>
            </div>
            """, unsafe_allow_html=True)

# =========================
# Historique
# =========================
if st.session_state.history:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <span class="section-tag" style="background:#0d1520;border-color:#1e3040;color:#3a5a6a">↑</span>
        <h2 class="section-title" style="font-style:italic">Historique de session</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"↳ {len(st.session_state.history)} génération(s)", expanded=False):
        for i, entry in enumerate(st.session_state.history):
            short_prompt = entry["prompt"][:120].replace("\n", " ") + ("…" if len(entry["prompt"]) > 120 else "")
            short_result = entry["result"][:200].replace("\n", " ") + ("…" if len(entry["result"]) > 200 else "")
            st.markdown(f"""
            <div class="history-entry">
                <div class="history-meta">#{len(st.session_state.history) - i:02d} · {entry['task']}</div>
                <div class="history-q">↳ {short_prompt}</div>
                <div class="history-a">{short_result}</div>
            </div>
            """, unsafe_allow_html=True)