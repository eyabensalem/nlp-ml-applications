import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Configuration de la page
# =========================
st.set_page_config(
    page_title="Word2Vec · Embeddings",
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

.block-container {
    padding: 2.5rem 3.5rem 5rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0a0c10 !important;
    border-right: 1px solid #1a1f2e !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown p {
    color: #6b7280 !important;
}
[data-testid="stSidebar"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    color: #4a5568 !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #1a1f2e !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #c07a3a !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    display: none;
}
.sidebar-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #c07a3a;
    margin-bottom: 0.3rem;
    display: block;
}
.sidebar-title {
    font-family: 'Fraunces', serif;
    font-size: 1rem;
    font-weight: 300;
    color: #e8e4dc;
    margin: 0 0 1.5rem 0;
}

/* ── Hero ─────────────────────────────────────── */
.hero {
    margin-bottom: 2.8rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #1a1f2e;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c07a3a;
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
    color: #d4935a;
}
.hero-sub {
    font-size: 0.88rem;
    color: #6b7280;
    font-weight: 300;
    max-width: 620px;
    line-height: 1.65;
}

/* ── Divider ─────────────────────────────────── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, #1a1f2e 0%, #3a2a1a 50%, #1a1f2e 100%);
    margin: 2.2rem 0;
    border: none;
}

/* ── Section ─────────────────────────────────── */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 1.1rem;
    margin-bottom: 0.7rem;
}
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #c07a3a;
    letter-spacing: 0.1em;
    background: #1a1206;
    border: 1px solid #3a2810;
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
    border-left: 2px solid #2a1e10;
    margin-bottom: 1.2rem;
}

/* ── Model badge ─────────────────────────────── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.35rem 0.9rem;
    border-radius: 3px;
    margin-bottom: 1rem;
}
.badge-cbow {
    color: #d4935a;
    background: #1a1206;
    border: 1px solid #3a2810;
    border-left: 3px solid #c07a3a;
}
.badge-skipgram {
    color: #8ab89a;
    background: #0f1a14;
    border: 1px solid #1e3328;
    border-left: 3px solid #5a8a6a;
}

/* ── Selectbox ───────────────────────────────── */
.stSelectbox label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a5568 !important;
}
.stSelectbox > div > div {
    background: #111318 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 5px !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.84rem !important;
}

/* ── Metrics ─────────────────────────────────── */
[data-testid="metric-container"] {
    background: #111318 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #3d4f62 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 1.7rem !important;
    font-weight: 300 !important;
    color: #d4935a !important;
}

/* ── Dataframes ──────────────────────────────── */
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
    background: #111318 !important;
    color: #c07a3a !important;
    border-bottom: 1px solid #1a1f2e !important;
    padding: 0.5rem 0.8rem !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) td {
    background: #0e1015 !important;
    color: #a0b4aa !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
    background: #111318 !important;
    color: #a0b4aa !important;
}

/* ── Expander ────────────────────────────────── */
[data-testid="stExpander"] {
    background: #111318 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    color: #6b7280 !important;
}

/* ── Corpus pills ────────────────────────────── */
.corpus-sample {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    padding: 0.8rem;
}
.corpus-line {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #8a7a6a;
    padding: 0.3rem 0.6rem;
    border-left: 2px solid #2a1e10;
}
.corpus-line span {
    color: #4a3a28;
    margin-right: 0.7rem;
}

/* ── Word selector zone ──────────────────────── */
.word-selector-wrap {
    background: #0e1015;
    border: 1px solid #1a1f2e;
    border-radius: 8px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.5rem;
}
.word-selector-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3d4f62;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.word-selector-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1f2e;
}

/* ── Col separator ───────────────────────────── */
.col-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4f62;
    margin-bottom: 0.7rem;
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

/* ── Alert ───────────────────────────────────── */
.stAlert {
    background: #111318 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
}
.stAlert p { color: #6b7280 !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# =========================
# Fonctions
# =========================
def generate_corpus(n_sentences: int = 5000) -> list:
    nouns = ["chat", "chien", "souris", "ballon", "table", "fromage", "arbre"]
    verbs = ["mange", "court", "dort", "saute", "joue", "chasse", "cache"]
    adjectives = ["petit", "grand", "rapide", "fort", "agile"]
    corpus = []
    for _ in range(n_sentences):
        sentence = (
            f"{random.choice(nouns)} {random.choice(verbs)} "
            f"{random.choice(nouns)} {random.choice(adjectives)}"
        )
        corpus.append(sentence)
    return corpus


def tokenize_corpus(corpus: list) -> list:
    return [sentence.split() for sentence in corpus]


@st.cache_resource
def train_word2vec_models(tokenized_docs_tuple, vector_size, window, epochs):
    tokenized_docs = list(tokenized_docs_tuple)
    cbow = Word2Vec(sentences=tokenized_docs, vector_size=vector_size, window=window,
                    min_count=1, sg=0, epochs=epochs, workers=1, seed=42)
    skipgram = Word2Vec(sentences=tokenized_docs, vector_size=vector_size, window=window,
                        min_count=1, sg=1, epochs=epochs, workers=1, seed=42)
    return cbow, skipgram


def vectors_to_dataframe(model):
    words = model.wv.index_to_key
    vectors = np.array([model.wv[word] for word in words])
    pca = PCA(n_components=3)
    v3d = pca.fit_transform(vectors)
    df = pd.DataFrame(v3d, columns=["x", "y", "z"])
    df["word"] = words
    return df


def cosine_sim(model, word_1, word_2):
    v1 = model.wv[word_1].reshape(1, -1)
    v2 = model.wv[word_2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def similar_words_to_dataframe(model, word, topn):
    similar = model.wv.most_similar(word, topn=topn)
    return pd.DataFrame(similar, columns=["Mot proche", "Similarité"])


def make_3d_fig(df, color, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"],
        mode="markers+text",
        text=df["word"],
        textfont=dict(family="DM Mono, monospace", size=11, color="#c9c4ba"),
        marker=dict(
            size=7,
            color=df["x"],
            colorscale=[[0, "#1a1206"], [0.5, color], [1, "#f0d0a0"]] if color == "#c07a3a"
                       else [[0, "#0f1a14"], [0.5, color], [1, "#a0e0b8"]],
            opacity=0.85,
            line=dict(width=0),
        ),
        hovertemplate="<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#0d0f14",
        margin=dict(l=0, r=0, t=36, b=0),
        height=420,
        title=dict(text=title, font=dict(family="DM Mono, monospace", size=11,
                   color="#4a5568"), x=0.01),
        scene=dict(
            bgcolor="#0d0f14",
            xaxis=dict(backgroundcolor="#0d0f14", gridcolor="#1a1f2e",
                       showbackground=True, zerolinecolor="#1a1f2e",
                       tickfont=dict(family="DM Mono", size=9, color="#3d4f62")),
            yaxis=dict(backgroundcolor="#0d0f14", gridcolor="#1a1f2e",
                       showbackground=True, zerolinecolor="#1a1f2e",
                       tickfont=dict(family="DM Mono", size=9, color="#3d4f62")),
            zaxis=dict(backgroundcolor="#0d0f14", gridcolor="#1a1f2e",
                       showbackground=True, zerolinecolor="#1a1f2e",
                       tickfont=dict(family="DM Mono", size=9, color="#3d4f62")),
        ),
        font=dict(family="DM Mono, monospace", color="#c9c4ba"),
    )
    return fig


# =========================
# Sidebar
# =========================
st.sidebar.markdown("""
<span class="sidebar-eyebrow">Hyperparamètres</span>
<h2 class="sidebar-title">Configurer le modèle</h2>
""", unsafe_allow_html=True)

n_sentences = st.sidebar.slider("Phrases générées", 1000, 10000, 5000, 1000)
vector_size  = st.sidebar.slider("Taille des vecteurs", 10, 100, 50, 10)
window       = st.sidebar.slider("Fenêtre de contexte", 1, 5, 3)
epochs       = st.sidebar.slider("Époques", 5, 50, 20, 5)
topn         = st.sidebar.slider("Mots voisins à afficher", 3, 10, 5)


# =========================
# Génération & entraînement
# =========================
random.seed(42)
np.random.seed(42)

docs = generate_corpus(n_sentences=n_sentences)
tokenized_docs = tokenize_corpus(docs)

cbow_model, skipgram_model = train_word2vec_models(
    tokenized_docs_tuple=tuple(tuple(s) for s in tokenized_docs),
    vector_size=vector_size,
    window=window,
    epochs=epochs
)

df_cbow     = vectors_to_dataframe(cbow_model)
df_skipgram = vectors_to_dataframe(skipgram_model)
vocabulary  = sorted(cbow_model.wv.index_to_key)


# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <span class="hero-eyebrow">Natural Language Processing · Représentations distribuées</span>
    <h1 class="hero-title">Word2Vec &amp; <em>similarité cosinus</em></h1>
    <p class="hero-sub">Entraînez deux architectures Word2Vec (CBOW et Skip-Gram), visualisez les embeddings projetés en 3D, et comparez la proximité sémantique entre les mots du corpus.</p>
</div>
""", unsafe_allow_html=True)


# =========================
# Aperçu corpus
# =========================
with st.expander("↳ Aperçu du corpus généré", expanded=False):
    lines_html = "".join(
        f'<div class="corpus-line"><span>·{i+1:02d}</span>{s}</div>'
        for i, s in enumerate(docs[:12])
    )
    st.markdown(f'<div class="corpus-sample">{lines_html}</div>', unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# =========================
# Sélection des mots
# =========================
st.markdown("""
<div class="section-header">
    <span class="section-tag">⌖</span>
    <h2 class="section-title">Sélection des mots à comparer</h2>
</div>
""", unsafe_allow_html=True)

col_w1, col_w2 = st.columns(2)
with col_w1:
    word_1 = st.selectbox("Mot A", vocabulary, index=0)
with col_w2:
    word_2 = st.selectbox("Mot B", vocabulary, index=1)

similarity_cbow     = cosine_sim(cbow_model,     word_1, word_2)
similarity_skipgram = cosine_sim(skipgram_model, word_1, word_2)

# Similarity bar
pct_cbow = int(similarity_cbow * 100)
pct_skip = int(similarity_skipgram * 100)
st.markdown(f"""
<div style="margin-top:1rem; display:flex; gap:1.5rem;">
  <div style="flex:1; background:#111318; border:1px solid #1a1f2e; border-radius:6px; padding:0.8rem 1.1rem;">
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; color:#4a5568; margin-bottom:0.5rem;">CBOW · cos({word_1}, {word_2})</div>
    <div style="background:#1a1206; border-radius:3px; height:6px; margin-bottom:0.4rem;">
      <div style="background:#c07a3a; width:{pct_cbow}%; height:6px; border-radius:3px; transition:width 0.4s;"></div>
    </div>
    <div style="font-family:'Fraunces',serif; font-size:1.6rem; font-weight:300; color:#d4935a;">{similarity_cbow:.4f}</div>
  </div>
  <div style="flex:1; background:#111318; border:1px solid #1a1f2e; border-radius:6px; padding:0.8rem 1.1rem;">
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; color:#4a5568; margin-bottom:0.5rem;">Skip-Gram · cos({word_1}, {word_2})</div>
    <div style="background:#0f1a14; border-radius:3px; height:6px; margin-bottom:0.4rem;">
      <div style="background:#5a8a6a; width:{pct_skip}%; height:6px; border-radius:3px; transition:width 0.4s;"></div>
    </div>
    <div style="font-family:'Fraunces',serif; font-size:1.6rem; font-weight:300; color:#8ab89a;">{similarity_skipgram:.4f}</div>
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# =========================
# Visualisation 3D + voisins
# =========================
st.markdown("""
<div class="section-header">
    <span class="section-tag">01</span>
    <h2 class="section-title">Projection 3D des embeddings</h2>
</div>
<p class="section-desc">Réduction dimensionnelle PCA → 3 composantes principales. Chaque point représente un mot du vocabulaire dans l'espace vectoriel appris.</p>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="model-badge badge-cbow">■ CBOW — Continuous Bag of Words</div>', unsafe_allow_html=True)
    st.plotly_chart(make_3d_fig(df_cbow, "#c07a3a", "embeddings · cbow · pca-3d"), use_container_width=True)

with col_right:
    st.markdown('<div class="model-badge badge-skipgram">■ Skip-Gram</div>', unsafe_allow_html=True)
    st.plotly_chart(make_3d_fig(df_skipgram, "#5a8a6a", "embeddings · skip-gram · pca-3d"), use_container_width=True)


st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# =========================
# Mots voisins
# =========================
st.markdown(f"""
<div class="section-header">
    <span class="section-tag">02</span>
    <h2 class="section-title">Top {topn} voisins de <em>{word_1}</em></h2>
</div>
<p class="section-desc">Mots les plus proches dans l'espace vectoriel selon la similarité cosinus, pour chaque architecture.</p>
""", unsafe_allow_html=True)

col_sim_l, col_sim_r = st.columns(2)

with col_sim_l:
    st.markdown('<div class="model-badge badge-cbow">CBOW</div>', unsafe_allow_html=True)
    df_sim_cbow = similar_words_to_dataframe(cbow_model, word_1, topn)
    st.dataframe(
        df_sim_cbow.style.background_gradient(subset=["Similarité"], cmap="Oranges"),
        use_container_width=True, hide_index=True
    )

with col_sim_r:
    st.markdown('<div class="model-badge badge-skipgram">Skip-Gram</div>', unsafe_allow_html=True)
    df_sim_sg = similar_words_to_dataframe(skipgram_model, word_1, topn)
    st.dataframe(
        df_sim_sg.style.background_gradient(subset=["Similarité"], cmap="Greens"),
        use_container_width=True, hide_index=True
    )


# =========================
# Résumé
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag" style="background:#0e1520;border-color:#1e2e40;color:#6a8ab8">∑</span>
    <h2 class="section-title" style="font-style:italic">Résumé comparatif</h2>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Phrases corpus", f"{n_sentences:,}")
c2.metric("Vocabulaire", len(vocabulary))
c3.metric("Sim. CBOW", f"{similarity_cbow:.4f}")
c4.metric("Sim. Skip-Gram", f"{similarity_skipgram:.4f}")