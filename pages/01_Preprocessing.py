import re
import streamlit as st
import spacy
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# =========================
# Configuration de la page
# =========================
st.set_page_config(
    page_title="NLP Pipeline · Préprocessing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CSS personnalisé
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #0d0f14 !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ─────────────────────── */
# footer, header { visibility: hidden; }
.block-container {
    padding: 3rem 4rem 4rem !important;
    max-width: 1280px !important;
}

/* ── Header ──────────────────────────────────────*/
.hero {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    margin-bottom: 3.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #1e2330;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5a8a6a;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 2.8rem;
    font-weight: 300;
    line-height: 1.1;
    color: #f0ece3;
    margin: 0;
}
.hero-title em {
    font-style: italic;
    color: #8ab89a;
}
.hero-sub {
    font-size: 0.9rem;
    color: #6b7280;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.6;
    margin-top: 0.4rem;
}

/* ── Step cards ──────────────────────────────── */
.step-header {
    display: flex;
    align-items: baseline;
    gap: 1.1rem;
    margin-bottom: 1.4rem;
}
.step-number {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #5a8a6a;
    letter-spacing: 0.1em;
    background: #0f1a14;
    border: 1px solid #1e3328;
    padding: 0.2rem 0.55rem;
    border-radius: 3px;
    flex-shrink: 0;
}
.step-title {
    font-family: 'Fraunces', serif;
    font-size: 1.35rem;
    font-weight: 300;
    color: #e8e4dc;
    margin: 0;
}
.step-desc {
    font-size: 0.84rem;
    color: #6b7280;
    line-height: 1.6;
    margin-bottom: 1.2rem;
    padding-left: 0;
    border-left: 2px solid #1e2330;
    padding-left: 1rem;
}

/* ── Text areas ───────────────────────────────── */
.stTextArea > div > div > textarea {
    background: #111318 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 6px !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.1rem !important;
    caret-color: #5a8a6a;
}
.stTextArea > div > div > textarea:focus {
    border-color: #2d4a38 !important;
    box-shadow: 0 0 0 3px rgba(90,138,106,0.08) !important;
    outline: none !important;
}
.stTextArea label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    color: #4a5568 !important;
    text-transform: uppercase !important;
}

/* ── Buttons ─────────────────────────────────── */
.stButton > button {
    background: #111318 !important;
    border: 1px solid #2d4a38 !important;
    color: #8ab89a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #0f1a14 !important;
    border-color: #5a8a6a !important;
    color: #c9e8d4 !important;
    box-shadow: 0 0 16px rgba(90,138,106,0.15) !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;
}

/* ── Alerts / info ───────────────────────────── */
.stAlert {
    background: #0e1117 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 6px !important;
    font-size: 0.83rem !important;
}
.stAlert p {
    color: #7a8099 !important;
}

/* ── Success ─────────────────────────────────── */
div[data-testid="stNotification"] {
    background: #0f1a14 !important;
    border-color: #2d4a38 !important;
}

/* ── Token pills display ─────────────────────── */
.token-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    padding: 1.1rem;
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 6px;
    min-height: 60px;
}
.token-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #a0c4ae;
    background: #0d1f15;
    border: 1px solid #1e3328;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    line-height: 1.5;
}
.token-pill.stem {
    color: #c4b5a0;
    background: #1a1510;
    border-color: #3a2e1e;
}
.token-pill.lemma {
    color: #a0b4c4;
    background: #0d1520;
    border-color: #1e2e3a;
}
.token-pill.filtered {
    color: #c4a0b4;
    background: #1a101a;
    border-color: #3a1e38;
}

/* ── Sentence cards ──────────────────────────── */
.sentence-card {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #8ab89a;
    background: #0d1f15;
    border: 1px solid #1e3328;
    border-left: 3px solid #3a8a5a;
    padding: 0.6rem 0.9rem;
    border-radius: 0 4px 4px 0;
    margin-bottom: 0.4rem;
    line-height: 1.6;
}

/* ── Divider ─────────────────────────────────── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, #1e2330 0%, #2d4a38 50%, #1e2330 100%);
    margin: 2.5rem 0;
    border: none;
}

/* ── Metrics ─────────────────────────────────── */
[data-testid="metric-container"] {
    background: #111318 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a5568 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 1.5rem !important;
    font-weight: 300 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] > div[style*="color: rgb(255"] {
    color: #8ab89a !important;
}

/* ── Column labels ───────────────────────────── */
.col-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d5246;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.col-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2330;
}

/* ── Warning ─────────────────────────────────── */
div[data-testid="stWarning"] {
    background: #181510 !important;
    border-color: #3a2e10 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Téléchargement ressources NLTK
# =========================
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk_resources()


# =========================
# Chargement du modèle spaCy
# =========================
@st.cache_resource
def load_spacy_model():
    return spacy.load("fr_core_news_sm")

try:
    nlp = load_spacy_model()
except OSError:
    st.error(
        "Le modèle spaCy `fr_core_news_sm` n'est pas installé.\n\n"
        "Commande : `python -m spacy download fr_core_news_sm`"
    )
    st.stop()


# =========================
# Fonctions utilitaires
# =========================
def clean_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ")
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-zàâäçéèêëîïôöùûüÿ'\-\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def tokenize_text(text: str):
    words = word_tokenize(text, language="french")
    sentences = sent_tokenize(text, language="french")
    return words, sentences

def apply_stemming(tokens: list) -> list:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def apply_lemmatization(text: str) -> list:
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_space]

def remove_stopwords(tokens: list) -> list:
    french_stopwords = set(stopwords.words("french"))
    return [t for t in tokens if t not in french_stopwords and len(t) > 1]

def render_pills(tokens, variant="default"):
    css_class = {"stem": "stem", "lemma": "lemma", "filtered": "filtered"}.get(variant, "")
    pills_html = "".join(
        f'<span class="token-pill {css_class}">{t}</span>' for t in tokens
    )
    return f'<div class="token-grid">{pills_html}</div>'


# =========================
# Données d'exemple
# =========================
DEFAULT_TEXT = """<p>Cette commune de quelque 500&nbsp;âmes peut déjà se targuer de faire partie&nbsp;des «&nbsp;Petites cités de caractère&nbsp;». Située dans le parc régional du Perche, elle présente un <strong>«&nbsp;ordonnancement orthogonal&nbsp;»</strong> typique des cités construites aux XVIII<sup>e</sup> et XIX<sup>e</sup>&nbsp;siècles, indique France Télévisions.</p>"""


# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <span class="hero-eyebrow">Natural Language Processing · Pipeline</span>
    <h1 class="hero-title">Pré-traitement des <em>données textuelles</em></h1>
    <p class="hero-sub">Nettoyage, tokenisation, stemming, lemmatisation et suppression des stop words — exécutez chaque étape séquentiellement.</p>
</div>
""", unsafe_allow_html=True)


# ─── Texte source ──────────────────────────────
st.markdown('<p class="col-label">Texte source</p>', unsafe_allow_html=True)
text = st.text_area(
    "Texte brut",
    DEFAULT_TEXT,
    height=160,
    label_visibility="collapsed",
    help="Collez un texte brut avec HTML, ponctuation ou caractères spéciaux."
)


# =========================
# Étape 1 — Nettoyage
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-header">
    <span class="step-number">01</span>
    <h2 class="step-title">Nettoyage</h2>
</div>
<p class="step-desc">Suppression des balises HTML et entités, mise en minuscules, élimination de la ponctuation et normalisation des espaces.</p>
""", unsafe_allow_html=True)

col_btn1, col_info1 = st.columns([1, 3])
with col_btn1:
    clean_clicked = st.button("→ Nettoyer", key="btn_clean")

if clean_clicked:
    st.session_state["clean_text"] = clean_text(text)

if "clean_text" in st.session_state:
    st.markdown('<p class="col-label">Résultat</p>', unsafe_allow_html=True)
    st.text_area("clean_result", st.session_state["clean_text"], height=90, disabled=True, label_visibility="collapsed")
else:
    with col_info1:
        st.info("Cliquez sur **→ Nettoyer** pour lancer cette étape.")


# =========================
# Étape 2 — Tokenisation
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-header">
    <span class="step-number">02</span>
    <h2 class="step-title">Tokenisation</h2>
</div>
<p class="step-desc">Découpage du texte nettoyé en unités atomiques : mots (tokens) et phrases.</p>
""", unsafe_allow_html=True)

col_btn2, col_info2 = st.columns([1, 3])
with col_btn2:
    tokenize_clicked = st.button("→ Tokeniser", key="btn_tok")

if tokenize_clicked:
    if "clean_text" not in st.session_state:
        st.warning("⚠ Nettoyez d'abord le texte (étape 01).")
    else:
        words, sentences = tokenize_text(st.session_state["clean_text"])
        st.session_state["tokenized_words"] = words
        st.session_state["tokenized_sentences"] = sentences

if "tokenized_words" in st.session_state:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="col-label">Mots</p>', unsafe_allow_html=True)
        st.markdown(render_pills(st.session_state["tokenized_words"]), unsafe_allow_html=True)
    with c2:
        st.markdown('<p class="col-label">Phrases</p>', unsafe_allow_html=True)
        for s in st.session_state["tokenized_sentences"]:
            st.markdown(f'<div class="sentence-card">{s}</div>', unsafe_allow_html=True)
elif "clean_text" in st.session_state:
    with col_info2:
        st.info("Texte nettoyé disponible. Cliquez sur **→ Tokeniser**.")


# =========================
# Étape 3 — Stemming & Lemmatisation
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-header">
    <span class="step-number">03</span>
    <h2 class="step-title">Stemming & Lemmatisation</h2>
</div>
<p class="step-desc">Deux approches de normalisation morphologique : réduction heuristique (stemming) et réduction à la forme canonique (lemmatisation via spaCy).</p>
""", unsafe_allow_html=True)

col_s, col_l = st.columns(2)
with col_s:
    stem_clicked = st.button("→ Stemming (Porter)", use_container_width=True)
with col_l:
    lemma_clicked = st.button("→ Lemmatisation (spaCy)", use_container_width=True)

if stem_clicked:
    if "tokenized_words" not in st.session_state:
        st.warning("⚠ Tokenisez d'abord le texte (étape 02).")
    else:
        st.session_state["stemmed_words"] = apply_stemming(st.session_state["tokenized_words"])

if lemma_clicked:
    if "clean_text" not in st.session_state:
        st.warning("⚠ Nettoyez d'abord le texte (étape 01).")
    else:
        st.session_state["lemmatized_words"] = apply_lemmatization(st.session_state["clean_text"])

res_col1, res_col2 = st.columns(2)
with res_col1:
    if "stemmed_words" in st.session_state:
        st.markdown('<p class="col-label">Racines (stems)</p>', unsafe_allow_html=True)
        st.markdown(render_pills(st.session_state["stemmed_words"], variant="stem"), unsafe_allow_html=True)
with res_col2:
    if "lemmatized_words" in st.session_state:
        st.markdown('<p class="col-label">Lemmes</p>', unsafe_allow_html=True)
        st.markdown(render_pills(st.session_state["lemmatized_words"], variant="lemma"), unsafe_allow_html=True)


# =========================
# Étape 4 — Stop words
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-header">
    <span class="step-number">04</span>
    <h2 class="step-title">Suppression des stop words</h2>
</div>
<p class="step-desc">Filtrage des mots grammaticaux à faible valeur informationnelle (articles, prépositions, conjonctions…) depuis les lemmes obtenus à l'étape précédente.</p>
""", unsafe_allow_html=True)

col_btn4, col_info4 = st.columns([1, 3])
with col_btn4:
    stop_clicked = st.button("→ Filtrer", key="btn_stop")

if stop_clicked:
    if "lemmatized_words" not in st.session_state:
        st.warning("⚠ Appliquez d'abord la lemmatisation (étape 03).")
    else:
        st.session_state["filtered_words"] = remove_stopwords(st.session_state["lemmatized_words"])

if "filtered_words" in st.session_state:
    before = len(st.session_state.get("lemmatized_words", []))
    after = len(st.session_state["filtered_words"])
    removed = before - after
    st.markdown(
        f'<p class="col-label">Tokens retenus — {after}/{before} &nbsp;·&nbsp; {removed} supprimés</p>',
        unsafe_allow_html=True
    )
    st.markdown(render_pills(st.session_state["filtered_words"], variant="filtered"), unsafe_allow_html=True)
elif "lemmatized_words" in st.session_state:
    with col_info4:
        st.info("Lemmes disponibles. Cliquez sur **→ Filtrer**.")


# =========================
# Résumé
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-header">
    <span class="step-number" style="background:#0f1520;border-color:#1e2e3a;color:#6a9ab8">∑</span>
    <h2 class="step-title" style="font-style:italic">État du pipeline</h2>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
def status(key): return "✓" if key in st.session_state else "—"
c1.metric("Nettoyage", status("clean_text"))
c2.metric("Tokenisation", status("tokenized_words"))
c3.metric("Lemmatisation", status("lemmatized_words"))
c4.metric("Stop words filtrés", status("filtered_words"))