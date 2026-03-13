import streamlit as st
import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)


# =========================
# Configuration page
# =========================
st.set_page_config(
    page_title="BERT · Analyse & QA",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    padding: 3rem 4rem 5rem !important;
    max-width: 1280px !important;
}

/* ── Hero ─────────────────────────────────────── */
.hero {
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #1f1a2e;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #8a6ac0;
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
    color: #a888d8;
}
.hero-sub {
    font-size: 0.88rem;
    color: #6b7280;
    font-weight: 300;
    max-width: 600px;
    line-height: 1.65;
}

/* ── Model chip ──────────────────────────────── */
.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    color: #7a5aaa;
    background: #130e1e;
    border: 1px solid #2a1e40;
    padding: 0.28rem 0.75rem;
    border-radius: 20px;
    margin-top: 0.9rem;
    margin-bottom: 0;
}
.model-chip::before {
    content: '◆';
    font-size: 0.5rem;
    color: #5a3a8a;
}

/* ── Divider ─────────────────────────────────── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, #1a1428 0%, #3a2460 50%, #1a1428 100%);
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
    color: #8a6ac0;
    letter-spacing: 0.1em;
    background: #130e1e;
    border: 1px solid #2a1e40;
    padding: 0.2rem 0.55rem;
    border-radius: 3px;
    flex-shrink: 0;
}
.section-title {
    font-family: 'Fraunces', serif;
    font-size: 1.35rem;
    font-weight: 300;
    color: #e8e4dc;
    margin: 0;
}
.section-desc {
    font-size: 0.84rem;
    color: #6b7280;
    line-height: 1.6;
    padding-left: 1rem;
    border-left: 2px solid #2a1e40;
    margin-bottom: 1.3rem;
}

/* ── Text areas ───────────────────────────────── */
.stTextArea > div > div > textarea {
    background: #111318 !important;
    border: 1px solid #1f1a2e !important;
    border-radius: 6px !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.75 !important;
    padding: 1rem 1.1rem !important;
    caret-color: #8a6ac0;
}
.stTextArea > div > div > textarea:focus {
    border-color: #3a2460 !important;
    box-shadow: 0 0 0 3px rgba(138,106,192,0.08) !important;
    outline: none !important;
}
.stTextArea label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #4a3a62 !important;
}

/* ── Text input ───────────────────────────────── */
.stTextInput > div > div > input {
    background: #111318 !important;
    border: 1px solid #1f1a2e !important;
    border-radius: 6px !important;
    color: #c9c4ba !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.84rem !important;
    padding: 0.65rem 1rem !important;
    caret-color: #8a6ac0;
}
.stTextInput > div > div > input:focus {
    border-color: #3a2460 !important;
    box-shadow: 0 0 0 3px rgba(138,106,192,0.08) !important;
    outline: none !important;
}
.stTextInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #4a3a62 !important;
}

/* ── Buttons ─────────────────────────────────── */
.stButton > button {
    background: #130e1e !important;
    border: 1px solid #3a2460 !important;
    color: #a888d8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 1.6rem !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #1a1030 !important;
    border-color: #6a40a0 !important;
    color: #c8a8f0 !important;
    box-shadow: 0 0 20px rgba(138,106,192,0.18) !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;
}

/* ── Metrics ─────────────────────────────────── */
[data-testid="metric-container"] {
    background: #111318 !important;
    border: 1px solid #1f1a2e !important;
    border-radius: 6px !important;
    padding: 1.1rem 1.3rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #3d3050 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 1.7rem !important;
    font-weight: 300 !important;
    color: #a888d8 !important;
}

/* ── Result card ─────────────────────────────── */
.result-card {
    background: #0f0c18;
    border: 1px solid #2a1e40;
    border-left: 3px solid #6a40a0;
    border-radius: 0 6px 6px 0;
    padding: 1.1rem 1.4rem;
    margin-top: 1.2rem;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a3a62;
    margin-bottom: 0.5rem;
}
.result-answer {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 300;
    font-style: italic;
    color: #c8a8f0;
    line-height: 1.3;
}
.result-snippet {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #8a6ac0;
    margin-top: 0.5rem;
    line-height: 1.6;
}

/* ── Sentiment badge ─────────────────────────── */
.sentiment-positive {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    color: #8ab89a;
    background: #0f1a14;
    border: 1px solid #2d4a38;
    border-left: 3px solid #5a8a6a;
    padding: 0.5rem 1rem;
    border-radius: 0 4px 4px 0;
}
.sentiment-negative {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    color: #c87878;
    background: #1a0f0f;
    border: 1px solid #4a2020;
    border-left: 3px solid #8a3a3a;
    padding: 0.5rem 1rem;
    border-radius: 0 4px 4px 0;
}

/* ── Confidence bar ──────────────────────────── */
.conf-wrap {
    margin-top: 0.9rem;
}
.conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a3a62;
    margin-bottom: 0.35rem;
}
.conf-track {
    background: #1a1428;
    border-radius: 3px;
    height: 5px;
    margin-bottom: 0.3rem;
    overflow: hidden;
}
.conf-fill {
    height: 5px;
    border-radius: 3px;
    background: linear-gradient(90deg, #3a2460, #a888d8);
}
.conf-value {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 300;
    color: #a888d8;
}

/* ── Alert ───────────────────────────────────── */
.stAlert {
    background: #0e0c18 !important;
    border: 1px solid #1f1a2e !important;
    border-radius: 6px !important;
}
.stAlert p { color: #6b7280 !important; font-size: 0.82rem !important; }

/* ── Expander ────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0e0c18 !important;
    border: 1px solid #1f1a2e !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    color: #5a4a78 !important;
}
[data-testid="stExpander"] .stMarkdown p,
[data-testid="stExpander"] .stMarkdown li {
    font-size: 0.85rem !important;
    color: #7a7090 !important;
    line-height: 1.7 !important;
}
[data-testid="stExpander"] .stMarkdown code {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #1a1428 !important;
    color: #a888d8 !important;
    padding: 0.15rem 0.4rem !important;
    border-radius: 3px !important;
}

/* ── Warning ─────────────────────────────────── */
div[data-testid="stWarning"] {
    background: #15120e !important;
    border-color: #3a2e10 !important;
}

/* ── Col label ───────────────────────────────── */
.col-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d3050;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.col-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1f1a2e;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Chargement des modèles
# =========================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_qa_components():
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

sentiment_model = load_sentiment_model()
qa_tokenizer, qa_model = load_qa_components()


# =========================
# Fonction QA
# =========================
def answer_question(question: str, context: str) -> dict:
    inputs = qa_tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_index = torch.argmax(outputs.start_logits, dim=1).item()
    end_index   = torch.argmax(outputs.end_logits,   dim=1).item()
    if end_index < start_index:
        end_index = start_index

    answer_ids = inputs["input_ids"][0][start_index:end_index + 1]
    answer = qa_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    start_score = torch.softmax(outputs.start_logits, dim=1)[0, start_index].item()
    end_score   = torch.softmax(outputs.end_logits,   dim=1)[0, end_index].item()
    confidence  = (start_score + end_score) / 2

    return {
        "answer": answer if answer else "Aucune réponse trouvée",
        "score": confidence,
    }


# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <span class="hero-eyebrow">Natural Language Processing · Transformers</span>
    <h1 class="hero-title">BERT &amp; <em>modèles pré-entraînés</em></h1>
    <p class="hero-sub">Deux tâches NLP illustrées avec DistilBERT : classification de sentiment (SST-2) et question-réponse extractive (SQuAD).</p>
    <span class="model-chip">distilbert-base-uncased · distilbert-base-cased-distilled-squad</span>
</div>
""", unsafe_allow_html=True)


# =========================
# Section 1 — Sentiment
# =========================
st.markdown("""
<div class="section-header">
    <span class="section-tag">01</span>
    <h2 class="section-title">Analyse de sentiment</h2>
</div>
<p class="section-desc">Classification binaire POSITIVE / NEGATIVE sur un texte libre. Le modèle a été fine-tuné sur SST-2 (Stanford Sentiment Treebank).</p>
""", unsafe_allow_html=True)

sentiment_text = st.text_area(
    "Texte à analyser",
    "I love natural language processing and machine learning.",
    height=110,
    key="sentiment_input"
)

col_btn_s, _ = st.columns([1, 3])
with col_btn_s:
    analyze_clicked = st.button("→ Analyser", key="btn_sentiment")

if analyze_clicked:
    if not sentiment_text.strip():
        st.warning("⚠ Veuillez entrer un texte.")
    else:
        with st.spinner("Inférence en cours…"):
            result = sentiment_model(sentiment_text)[0]
        st.session_state["sentiment_result"] = result

if "sentiment_result" in st.session_state:
    r = st.session_state["sentiment_result"]
    label = r["label"]
    score = r["score"]
    pct   = int(score * 100)

    badge_class = "sentiment-positive" if label == "POSITIVE" else "sentiment-negative"
    icon        = "↑" if label == "POSITIVE" else "↓"

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(f'<div class="{badge_class}">{icon} {label}</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-label">Confiance du modèle</div>
            <div class="conf-track"><div class="conf-fill" style="width:{pct}%"></div></div>
            <div class="conf-value">{score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)


# =========================
# Section 2 — QA
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag">02</span>
    <h2 class="section-title">Question-réponse extractive</h2>
</div>
<p class="section-desc">Le modèle lit un contexte et extrait le span de texte qui répond le mieux à la question posée. Fine-tuné sur SQuAD.</p>
""", unsafe_allow_html=True)

col_ctx, col_q = st.columns([3, 2])

with col_ctx:
    st.markdown('<p class="col-label">Contexte</p>', unsafe_allow_html=True)
    context = st.text_area(
        "Contexte",
        "Paris est la capitale de la France. Elle est connue pour la Tour Eiffel.",
        height=140,
        key="qa_context",
        label_visibility="collapsed"
    )

with col_q:
    st.markdown('<p class="col-label">Question</p>', unsafe_allow_html=True)
    question = st.text_input(
        "Question",
        "Quelle est la capitale de la France ?",
        key="qa_question",
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    qa_clicked = st.button("→ Trouver la réponse", key="btn_qa")

if qa_clicked:
    if not context.strip() or not question.strip():
        st.warning("⚠ Veuillez remplir le contexte et la question.")
    else:
        with st.spinner("Extraction en cours…"):
            qa_result = answer_question(question, context)
        st.session_state["qa_result"] = qa_result

if "qa_result" in st.session_state:
    qr    = st.session_state["qa_result"]
    score = qr["score"]
    pct   = int(score * 100)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Réponse extraite</div>
        <div class="result-answer">« {qr['answer']} »</div>
        <div class="conf-wrap" style="margin-top:0.9rem">
            <div class="conf-label">Score de confiance</div>
            <div class="conf-track"><div class="conf-fill" style="width:{pct}%"></div></div>
            <div class="conf-value">{score:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# Explication
# =========================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

with st.expander("↳ Comment ça fonctionne — BERT & QA extractive"):
    st.markdown("""
**BERT** (Bidirectional Encoder Representations from Transformers) est un modèle Transformer bidirectionnel pré-entraîné sur de larges corpus textuels.

**Analyse de sentiment** — le modèle utilisé est `distilbert-base-uncased`, fine-tuné sur SST-2. Il produit une classification binaire avec un score de confiance.

**Question-réponse extractive** — `distilbert-base-cased-distilled-squad` prédit les positions de début et de fin du span répondant à la question dans le contexte fourni. Il ne génère pas de texte : il *extrait*.

Le score de confiance affiché est la moyenne des probabilités softmax sur les logits de début et de fin de span.
    """)