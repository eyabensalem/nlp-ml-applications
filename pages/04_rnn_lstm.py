import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


# =========================
# Configuration de la page
# =========================
st.set_page_config(
    page_title="RNN & LSTM · Prédiction séquentielle",
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
[data-testid="stSidebar"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.07em !important;
    color: #4a5568 !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #111318 !important;
    border: 1px solid #1e2540 !important;
    border-radius: 5px !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #111318 !important;
    border: 1px solid #1e2540 !important;
    border-radius: 5px !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    background: #111318 !important;
    border: 1px solid #1e2540 !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #7a5ab8 !important;
}
.sidebar-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #9a7ad4;
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
    color: #9a7ad4;
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
.hero-title em { font-style: italic; color: #b89ae0; }
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
    background: linear-gradient(90deg, #1a1f2e 0%, #2e1e4a 50%, #1a1f2e 100%);
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
    color: #9a7ad4;
    letter-spacing: 0.1em;
    background: #160e24;
    border: 1px solid #2e1e4a;
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
    border-left: 2px solid #2e1e4a;
    margin-bottom: 1.2rem;
}

/* ── Word pills ───────────────────────────────── */
.word-sequence {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    padding: 1rem;
    background: #111318;
    border: 1px solid #1a1f2e;
    border-radius: 6px;
}
.word-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #b89ae0;
    background: #160e24;
    border: 1px solid #2e1e4a;
    padding: 0.22rem 0.65rem;
    border-radius: 3px;
}
.word-pill-index {
    font-size: 0.6rem;
    color: #4a3a68;
    margin-right: 0.3rem;
    vertical-align: super;
}
.vocab-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #8a8a9a;
    background: #111318;
    border: 1px solid #1e2030;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
}
.vocab-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    padding: 0.8rem 1rem;
    background: #111318;
    border: 1px solid #1a1f2e;
    border-radius: 6px;
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
    color: #b89ae0 !important;
}

/* ── Success ─────────────────────────────────── */
[data-testid="stNotification"] {
    background: #120e1e !important;
    border: 1px solid #2e1e4a !important;
    border-left: 3px solid #7a5ab8 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Step cards ──────────────────────────────── */
.step-card {
    background: #0e1015;
    border: 1px solid #1a1f2e;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.step-card-header {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1a1f2e;
    flex-wrap: wrap;
}
.step-number-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #9a7ad4;
    background: #160e24;
    border: 1px solid #2e1e4a;
    padding: 0.25rem 0.6rem;
    border-radius: 3px;
}
.step-input-word {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 300;
    color: #e8e4dc;
}
.step-input-word em { font-style: italic; color: #b89ae0; }
.predicted-badge {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: #7ad4b0;
    background: #0e1e18;
    border: 1px solid #1e4a38;
    padding: 0.3rem 0.8rem;
    border-radius: 3px;
}

/* ── Sub-label ───────────────────────────────── */
.sub-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.67rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4562;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sub-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1f2e;
}

/* ── Code ────────────────────────────────────── */
[data-testid="stCode"] {
    background: #0a0c10 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 5px !important;
}
[data-testid="stCode"] code {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.76rem !important;
    color: #9a7ad4 !important;
}

/* ── Dataframe ───────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="stDataFrame"] thead th {
    background: #111318 !important;
    color: #9a7ad4 !important;
    border-bottom: 1px solid #1a1f2e !important;
    padding: 0.5rem 0.7rem !important;
}
[data-testid="stDataFrame"] tbody td {
    color: #a090c0 !important;
    border-color: #1a1f2e !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) td { background: #0e1015 !important; }
[data-testid="stDataFrame"] tbody tr:nth-child(even) td { background: #111318 !important; }

/* ── Expander ────────────────────────────────── */
[data-testid="stExpander"] {
    background: #111318 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #6b7280 !important;
}
[data-testid="stExpander"] p, [data-testid="stExpander"] li {
    font-size: 0.85rem !important;
    color: #7a8090 !important;
    line-height: 1.7 !important;
}
[data-testid="stExpander"] strong { color: #b89ae0 !important; }
[data-testid="stExpander"] code {
    font-family: 'DM Mono', monospace !important;
    color: #9a7ad4 !important;
    background: #160e24 !important;
    padding: 0.1rem 0.4rem !important;
    border-radius: 3px !important;
}

/* ── Col label ───────────────────────────────── */
.col-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4562;
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
</style>
""", unsafe_allow_html=True)


# =========================
# Reproductibilité
# =========================
torch.manual_seed(42)
np.random.seed(42)


# =========================
# Fonctions
# =========================
def build_vocabulary(sentence: str):
    words = sentence.split()
    vocab = sorted(list(set(words)))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return words, vocab, word_to_index, index_to_word


def build_training_data(words, word_to_index):
    x = torch.tensor([[word_to_index[w] for w in words[:-1]]], dtype=torch.long)
    y = torch.tensor([word_to_index[w] for w in words[1:]], dtype=torch.long)
    return x, y


class SimpleSequenceModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, model_type="RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if model_type == "RNN":
            self.sequence_layer = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        else:
            self.sequence_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.model_type = model_type

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.sequence_layer(x, hidden)
        return self.output_layer(output), hidden


def train_model(model, x_tensor, y_tensor, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(x_tensor)
        loss = criterion(logits[0], y_tensor)
        loss.backward()
        optimizer.step()
    return float(loss.item())


def run_inference(model, x_tensor, input_words, vocab, model_type):
    results = []
    hidden = None
    with torch.no_grad():
        for step in range(x_tensor.shape[1]):
            x_t = x_tensor[:, step:step + 1]
            logits_t, hidden = model(x_t, hidden)
            if model_type == "LSTM":
                h = hidden[0][0, 0].cpu().numpy()
                c = hidden[1][0, 0].cpu().numpy()
            else:
                h = hidden[0, 0].cpu().numpy()
                c = None
            probs = F.softmax(logits_t[0, 0], dim=-1).cpu().numpy()
            results.append({
                "step": step + 1,
                "input_word": input_words[step],
                "hidden_state": h,
                "cell_state": c,
                "probabilities": probs,
                "predicted_word": vocab[int(np.argmax(probs))],
            })
    return results


# =========================
# Sidebar
# =========================
st.sidebar.markdown("""
<span class="sidebar-eyebrow">Architecture séquentielle</span>
<h2 class="sidebar-title">Paramètres du modèle</h2>
""", unsafe_allow_html=True)

sentence      = st.sidebar.text_input("Phrase d'exemple", value="Le chat mange la souris")
model_type    = st.sidebar.selectbox("Architecture", ["RNN", "LSTM"])
embedding_dim = st.sidebar.slider("Embedding dim", 2, 64, 8, step=2)
hidden_dim    = st.sidebar.slider("Hidden dim", 2, 128, 16, step=2)
epochs        = st.sidebar.slider("Époques", 10, 2000, 300, step=10)
learning_rate = st.sidebar.number_input("Learning rate", 0.0001, 1.0, 0.05, step=0.01, format="%.4f")


# =========================
# Données
# =========================
words, vocab, word_to_index, index_to_word = build_vocabulary(sentence)

if len(words) < 2:
    st.error("La phrase doit contenir au moins deux mots.")
    st.stop()

x_tensor, y_tensor = build_training_data(words, word_to_index)
vocab_size = len(vocab)


# =========================
# HERO
# =========================
st.markdown(f"""
<div class="hero">
    <span class="hero-eyebrow">Natural Language Processing · Modèles séquentiels</span>
    <h1 class="hero-title">RNN &amp; <em>LSTM</em></h1>
    <p class="hero-sub">Entraînez un réseau récurrent ({model_type}) sur une phrase, observez l'évolution des états cachés à chaque pas de temps et inspectez la distribution de probabilité sur le vocabulaire.</p>
</div>
""", unsafe_allow_html=True)


# =========================
# Phrase & Vocabulaire
# =========================
st.markdown("""
<div class="section-header">
    <span class="section-tag">⌗</span>
    <h2 class="section-title">Phrase &amp; vocabulaire</h2>
</div>
""", unsafe_allow_html=True)

col_phrase, col_vocab = st.columns(2)

with col_phrase:
    st.markdown('<p class="col-label">Séquence d\'entrée</p>', unsafe_allow_html=True)
    pills = "".join(
        f'<span class="word-pill"><span class="word-pill-index">{i}</span>{w}</span>'
        for i, w in enumerate(words)
    )
    st.markdown(f'<div class="word-sequence">{pills}</div>', unsafe_allow_html=True)

with col_vocab:
    st.markdown(f'<p class="col-label">Vocabulaire — {vocab_size} tokens</p>', unsafe_allow_html=True)
    vpills = "".join(f'<span class="vocab-pill">{w}</span>' for w in vocab)
    st.markdown(f'<div class="vocab-grid">{vpills}</div>', unsafe_allow_html=True)


# =========================
# Entraînement
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

model = SimpleSequenceModel(vocab_size, embedding_dim, hidden_dim, model_type)
final_loss = train_model(model, x_tensor, y_tensor, epochs, float(learning_rate))

st.success(f"Entraînement terminé — loss finale : **{final_loss:.4f}** · {epochs} époques · lr = {learning_rate}")


# =========================
# Résumé architecture
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag">01</span>
    <h2 class="section-title">Architecture</h2>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Architecture", model_type)
c2.metric("Vocabulaire", vocab_size)
c3.metric("Embedding dim", embedding_dim)
c4.metric("Hidden dim", hidden_dim)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("↳ RNN vs LSTM — notes théoriques", expanded=False):
    st.markdown("""
**RNN** — transmet un seul état caché `h_t` d'un pas de temps au suivant. Simple et rapide, mais sujet à la disparition du gradient sur les longues séquences.

**LSTM** — ajoute un état mémoire `c_t` contrôlé par trois portes (*input*, *forget*, *output*). Conserve ou efface sélectivement l'information sur de longues dépendances.

Notations :
- `h_t` — état caché au pas de temps *t*
- `c_t` — mémoire interne, propre au LSTM
- `softmax(logits)` — distribution de probabilité sur le vocabulaire
    """)


# =========================
# Inférence pas à pas
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag">02</span>
    <h2 class="section-title">Inférence pas à pas</h2>
</div>
<p class="section-desc">À chaque pas de temps, le modèle lit un token, met à jour son état caché et produit une distribution de probabilité sur le vocabulaire.</p>
""", unsafe_allow_html=True)

step_results = run_inference(model, x_tensor, words, vocab, model_type)

for result in step_results:
    st.markdown(f"""
<div class="step-card">
  <div class="step-card-header">
    <span class="step-number-badge">t = {result['step']:02d}</span>
    <span class="step-input-word">entrée → <em>{result['input_word']}</em></span>
    <span class="predicted-badge">→ prédit : {result['predicted_word']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    col_states, col_probs = st.columns([1, 2])

    with col_states:
        st.markdown('<p class="sub-label">État caché h_t</p>', unsafe_allow_html=True)
        st.code(np.array2string(np.round(result["hidden_state"], 3), separator=", "), language="python")
        if model_type == "LSTM" and result["cell_state"] is not None:
            st.markdown('<p class="sub-label">Mémoire c_t</p>', unsafe_allow_html=True)
            st.code(np.array2string(np.round(result["cell_state"], 3), separator=", "), language="python")

    with col_probs:
        st.markdown('<p class="sub-label">Distribution sur le vocabulaire</p>', unsafe_allow_html=True)
        prob_df = pd.DataFrame([result["probabilities"]], columns=vocab, index=["P(·)"])
        st.dataframe(
            prob_df.style.background_gradient(cmap="Purples").format("{:.4f}"),
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)


# =========================
# Résumé
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag" style="background:#0e1520;border-color:#1e2e40;color:#6a8ab8">∑</span>
    <h2 class="section-title" style="font-style:italic">Résumé</h2>
</div>
""", unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Modèle", model_type)
s2.metric("Pas de temps", len(step_results))
s3.metric("Loss finale", f"{final_loss:.4f}")
s4.metric("Tokens vocab", vocab_size)