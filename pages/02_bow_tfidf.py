import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# =========================
# Configuration de la page
# =========================
st.set_page_config(
    page_title="Vectorisation · BoW & TF-IDF",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================
# CSS personnalisé
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

.block-container {
    padding: 3rem 4rem 5rem !important;
    max-width: 1320px !important;
}

/* ── Hero ─────────────────────────────────────── */
.hero {
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #1a1f2e;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5a7a9e;
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
.hero-title em {
    font-style: italic;
    color: #7aaac8;
}
.hero-sub {
    font-size: 0.88rem;
    color: #6b7280;
    font-weight: 300;
    max-width: 580px;
    line-height: 1.65;
}

/* ── Input zone ───────────────────────────────── */
.input-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4f62;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.input-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1f2e;
}

.stTextArea > div > div > textarea {
    background: #111318 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.84rem !important;
    line-height: 1.75 !important;
    padding: 1rem 1.1rem !important;
    caret-color: #5a7a9e;
}
.stTextArea > div > div > textarea:focus {
    border-color: #2a3d54 !important;
    box-shadow: 0 0 0 3px rgba(90,122,158,0.08) !important;
    outline: none !important;
}
.stTextArea label {
    display: none !important;
}

/* ── Divider ─────────────────────────────────── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, #1a1f2e 0%, #2a3d54 50%, #1a1f2e 100%);
    margin: 2.5rem 0;
    border: none;
}

/* ── Section headers ──────────────────────────── */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 1.1rem;
    margin-bottom: 0.8rem;
}
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #5a7a9e;
    letter-spacing: 0.1em;
    background: #0e141c;
    border: 1px solid #1e2e40;
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
    border-left: 2px solid #1a1f2e;
    margin-bottom: 1.2rem;
}

/* ── Dataframe ────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}
[data-testid="stDataFrame"] thead th {
    background: #0e141c !important;
    color: #5a7a9e !important;
    font-weight: 400 !important;
    border-bottom: 1px solid #1a1f2e !important;
    padding: 0.55rem 0.8rem !important;
    text-align: center !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background: #111318 !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background: #0e1015 !important;
}
[data-testid="stDataFrame"] tbody td {
    color: #a0b8cc !important;
    border-color: #1a1f2e !important;
    text-align: center !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: #0e1a28 !important;
}

/* ── Metrics ─────────────────────────────────── */
[data-testid="metric-container"] {
    background: #111318 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
    padding: 1.1rem 1.3rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #3d4f62 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 2rem !important;
    font-weight: 300 !important;
    color: #7aaac8 !important;
}

/* ── Bar chart ────────────────────────────────── */
[data-testid="stVegaLiteChart"] {
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
    background: #111318 !important;
    padding: 1rem !important;
}

/* ── Alert / info ─────────────────────────────── */
.stAlert {
    background: #0e141c !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
}
.stAlert p {
    color: #6b7280 !important;
    font-size: 0.83rem !important;
}

/* ── Warning ─────────────────────────────────── */
div[data-testid="stWarning"] {
    background: #15120e !important;
    border-color: #3a2e10 !important;
}

/* ── Heatmap legend ───────────────────────────── */
.legend-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 1rem;
}
.legend-swatch {
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #0e141c 0%, #1e3a52 40%, #2a5a80 70%, #7aaac8 100%);
}
.legend-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3d4f62;
    margin-top: 0.2rem;
}

/* ── Col label ────────────────────────────────── */
.col-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4f62;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.col-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1f2e;
}

/* ── Doc pills ────────────────────────────────── */
.doc-list {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    margin-bottom: 1.5rem;
}
.doc-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #7aaac8;
    background: #0e141c;
    border: 1px solid #1e2e40;
    border-left: 3px solid #2a5a80;
    padding: 0.45rem 0.9rem;
    border-radius: 0 4px 4px 0;
}
.doc-index {
    color: #3d4f62;
    margin-right: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Fonctions
# =========================
def split_documents(text: str):
    return [doc.strip() for doc in text.split("\n") if doc.strip()]

def compute_bow_matrix(docs):
    vec = CountVectorizer()
    m = vec.fit_transform(docs)
    return pd.DataFrame(m.toarray(), columns=vec.get_feature_names_out(),
                        index=[f"Doc {i+1}" for i in range(len(docs))])

def compute_tf_matrix(docs):
    vec = TfidfVectorizer(use_idf=False, norm="l1")
    m = vec.fit_transform(docs)
    return pd.DataFrame(m.toarray().round(4), columns=vec.get_feature_names_out(),
                        index=[f"Doc {i+1}" for i in range(len(docs))])

def compute_tfidf_matrix(docs):
    vec = TfidfVectorizer()
    m = vec.fit_transform(docs)
    return pd.DataFrame(m.toarray().round(4), columns=vec.get_feature_names_out(),
                        index=[f"Doc {i+1}" for i in range(len(docs))])


# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <span class="hero-eyebrow">Natural Language Processing · Représentation vectorielle</span>
    <h1 class="hero-title">Bag of Words &amp; <em>TF-IDF</em></h1>
    <p class="hero-sub">Deux méthodes classiques pour transformer le texte en vecteurs numériques : comptage brut (BoW), fréquence relative (TF) et pondération par rareté (TF-IDF).</p>
</div>
""", unsafe_allow_html=True)


# =========================
# Input
# =========================
DEFAULT_TEXT = "J'aime le NLP.\nLe NLP est passionnant.\nLe NLP et l'IA sont liés."

st.markdown('<p class="col-label">Corpus — une phrase par ligne</p>', unsafe_allow_html=True)
DOCUMENT = st.text_area("corpus", DEFAULT_TEXT, height=140, label_visibility="collapsed")

if not DOCUMENT:
    st.stop()

docs = split_documents(DOCUMENT)

if len(docs) == 0:
    st.warning("⚠ Entrez au moins une phrase.")
    st.stop()

# Show parsed docs
pills_html = "".join(
    f'<div class="doc-pill"><span class="doc-index">doc·{i+1:02d}</span>{d}</div>'
    for i, d in enumerate(docs)
)
st.markdown(f'<div class="doc-list">{pills_html}</div>', unsafe_allow_html=True)


# =========================
# BoW
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag">01</span>
    <h2 class="section-title">Bag of Words</h2>
</div>
<p class="section-desc">Chaque document est représenté par le <strong>nombre brut d'occurrences</strong> de chaque mot du vocabulaire. L'ordre des mots est ignoré.</p>
""", unsafe_allow_html=True)

bow_df = compute_bow_matrix(docs)
st.dataframe(bow_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)


# =========================
# TF
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag">02</span>
    <h2 class="section-title">Term Frequency (TF)</h2>
</div>
<p class="section-desc">Fréquence <strong>relative</strong> de chaque terme dans le document (normalisée L1). Permet de comparer des documents de longueurs différentes.</p>
""", unsafe_allow_html=True)

tf_df = compute_tf_matrix(docs)
st.dataframe(tf_df.style.background_gradient(cmap="Blues", axis=None).format("{:.4f}"), use_container_width=True)


# =========================
# TF-IDF
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag">03</span>
    <h2 class="section-title">TF-IDF</h2>
</div>
<p class="section-desc">TF × IDF — les mots <strong>rares et discriminants</strong> reçoivent un poids plus élevé. Les termes très fréquents dans tout le corpus sont pénalisés.</p>
""", unsafe_allow_html=True)

tfidf_df = compute_tfidf_matrix(docs)
st.dataframe(tfidf_df.style.background_gradient(cmap="Blues", axis=None).format("{:.4f}"), use_container_width=True)


# =========================
# Résumé & graphique
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag" style="background:#0e1520;border-color:#1e2e40;color:#6a8ab8">∑</span>
    <h2 class="section-title" style="font-style:italic">Résumé du corpus</h2>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Documents", len(docs))
c2.metric("Vocabulaire", bow_df.shape[1])
c3.metric("Occurrences totales", int(bow_df.sum().sum()))

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="col-label">Fréquence par terme (BoW)</p>', unsafe_allow_html=True)

freq_df = bow_df.sum().sort_values(ascending=False).reset_index()
freq_df.columns = ["terme", "occurrences"]
st.bar_chart(freq_df.set_index("terme"), color="#2a5a80", height=220)