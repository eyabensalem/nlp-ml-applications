import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from mistralai import Mistral
from dotenv import load_dotenv

# --- Données d'exemple ---
documents = [
    {
        "title": "Chatbot Éducatif avec RASA",
        "description": "Un chatbot basé sur RASA pour répondre aux questions des étudiants en master sur leurs cours. Idéal pour les projets de NLP.",
        "tags": ["NLP", "Éducation", "Python"],
        "lien": "https://rasa.com/docs/rasa/"
    },
    {
        "title": "Détection de Fraudes avec PyTorch",
        "description": "Modèle de détection de fraudes bancaires utilisant PyTorch et des techniques de Data Science. Parfait pour un mémoire en finance ou sécurité.",
        "tags": ["Data Science", "Sécurité", "PyTorch"],
        "lien": "https://pytorch.org/tutorials/"
    },
    {
        "title": "Système de Recommandation pour les MOOCs",
        "description": "Un système de recommandation de cours en ligne (MOOCs) basé sur les préférences des étudiants. Utilise Scikit-learn et Pandas.",
        "tags": ["Data Science", "Recommandation", "Scikit-learn"],
        "lien": "https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble"
    },
    {
        "title": "Monitoring de la Qualité de l'Air avec Raspberry Pi",
        "description": "Projet IoT pour surveiller la qualité de l'air dans une salle de classe. Utilise des capteurs, MQTT et une base de données InfluxDB.",
        "tags": ["IoT", "Raspberry Pi", "MQTT"],
        "lien": "https://projects.raspberrypi.org/en/projects/air-quality"
    },
    {
        "title": "Certification Décentralisée avec Blockchain",
        "description": "Plateforme de certification des diplômes utilisant la blockchain Ethereum et des smart contracts en Solidity.",
        "tags": ["Blockchain", "Solidity", "Ethereum"],
        "lien": "https://docs.soliditylang.org/"
    }
]

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)


def main():
    st.set_page_config(page_title="RAG — Recherche de projets", layout="wide")

    # ══════════════════════════════════════════════
    # THÈME — Futuriste Clair
    # ══════════════════════════════════════════════
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;600;800;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', system-ui, sans-serif;
    color: #0f172a;
}

.stApp {
    background:
        radial-gradient(ellipse 80% 40% at 15% 0%, rgba(14,165,233,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 30% at 85% 0%, rgba(99,102,241,0.07) 0%, transparent 55%),
        linear-gradient(180deg, #eef3f9 0%, #f0f4f8 40%, #edf2f7 100%);
    min-height: 100vh;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ── TITRE ── */
.main-title {
    text-align: center;
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: 0.15em;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #0ea5e9 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    margin-bottom: 0.2rem;
    filter: drop-shadow(0 0 18px rgba(14,165,233,0.2));
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.subtitle {
    text-align: center;
    font-family: 'DM Sans', sans-serif;
    font-style: italic;
    color: #64748b;
    font-size: 1rem;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}

.divider-line {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin: 0.5rem auto 1.8rem auto;
}
.divider-line::before,
.divider-line::after {
    content: '';
    flex: 1;
    max-width: 200px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #0ea5e9);
}
.divider-line::after {
    background: linear-gradient(270deg, transparent, #0ea5e9);
}
.divider-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #0ea5e9;
    box-shadow: 0 0 10px #0ea5e9;
}

/* ── PANELS ── */
.panel {
    background: #ffffff;
    border: 1px solid #c8d8e8;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 8px 24px rgba(14,165,233,0.07);
    position: relative;
    margin-bottom: 16px;
    overflow: hidden;
    animation: fadeUp 0.4s ease backwards;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
    border-radius: 16px 16px 0 0;
}

.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    color: #0ea5e9;
    margin-bottom: 1rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── CARTE PROJET ── */
.project-card {
    background: #ffffff;
    border: 1px solid #c8d8e8;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.4s ease backwards;
}
.project-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #0ea5e9, #6366f1);
    border-radius: 4px 0 0 4px;
}
.project-card:hover {
    box-shadow: 0 6px 22px rgba(14,165,233,0.14);
    transform: translateY(-2px);
}
.project-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 6px;
    letter-spacing: 0.04em;
}
.project-desc {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.93rem;
    color: #475569;
    line-height: 1.6;
    margin-bottom: 10px;
}
.tag-pill {
    display: inline-block;
    background: linear-gradient(135deg, #e0f2fe, #ede9fe);
    color: #0369a1;
    border: 1px solid #7dd3fc;
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 0.74rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.05em;
    margin-right: 6px;
    margin-bottom: 4px;
}
.project-link {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.84rem;
    font-weight: 600;
    color: #0ea5e9;
    text-decoration: none;
    letter-spacing: 0.04em;
}

/* ── RÉPONSE LLM ── */
.llm-response {
    background: linear-gradient(145deg, #f0f9ff, #f5f3ff);
    border: 1.5px solid #a5b4fc;
    border-left: 4px solid #6366f1;
    border-radius: 14px;
    padding: 20px 22px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.97rem;
    color: #1e1b4b;
    line-height: 1.7;
    box-shadow: 0 4px 16px rgba(99,102,241,0.10);
    margin-bottom: 16px;
}

/* ── INPUT ── */
.stTextInput > label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #c8d8e8 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;
    font-style: italic;
}

/* ── SPINNER ── */
div[data-testid="stSpinner"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #0ea5e9 !important;
}

/* ── EXPANDER ── */
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #c8d8e8 !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04) !important;
    overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    padding: 12px 16px !important;
}
div[data-testid="stExpander"] summary:hover {
    background: #f0f9ff !important;
}

/* ── SUBHEADER ── */
h2, h3 {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin-bottom: 1rem !important;
}

/* ── HR ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #c8d8e8, transparent);
    margin: 1.4rem 0;
}

/* ── ALERTS ── */
.stInfo {
    background: #f0f9ff !important;
    border: 1.5px solid #7dd3fc !important;
    border-left: 4px solid #0ea5e9 !important;
    color: #0c4a6e !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── CAPTION ── */
div[data-testid="stCaptionContainer"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-style: italic !important;
    color: #64748b !important;
    font-size: 0.87rem !important;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: none; }
}
</style>
""", unsafe_allow_html=True)

    # ── Titre ──
    st.markdown('<div class="main-title">◈ RAG PROJETS ◈</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Recherche sémantique d\'idées de projets augmentée par LLM.</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="divider-line"><span class="divider-dot"></span></div>', unsafe_allow_html=True)

    index, embed_model = build_faiss_index(documents)

    # ── Barre de recherche ──
    query = st.text_input(
        "Recherche de projet",
        placeholder="ex : projet NLP, IoT pour l'éducation, blockchain..."
    )

    if query:
        with st.spinner("⬡ Recherche en cours..."):
            retrieved_docs = retrieve_docs(query, index, embed_model, documents)
            response = generate_response(query, retrieved_docs)

        # Réponse LLM
        st.markdown('<div class="section-title">⬡ Réponse de l\'assistant</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="llm-response">{response}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Documents sources
        st.markdown('<div class="section-title">⬡ Documents utilisés</div>', unsafe_allow_html=True)
        for doc in retrieved_docs:
            tags_html = "".join(f'<span class="tag-pill">{t}</span>' for t in doc["tags"])
            st.markdown(
                f'<div class="project-card">'
                f'<div class="project-title">{doc["title"]}</div>'
                f'<div class="project-desc">{doc["description"]}</div>'
                f'<div style="margin-bottom:10px">{tags_html}</div>'
                f'<a class="project-link" href="{doc["lien"]}" target="_blank">↗ Voir la documentation</a>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

    # ── Bibliothèque complète ──
    st.markdown('<div class="section-title">⬡ Tous les projets disponibles</div>', unsafe_allow_html=True)
    for doc in documents:
        tags_html = "".join(f'<span class="tag-pill">{t}</span>' for t in doc["tags"])
        with st.expander(doc["title"]):
            st.markdown(
                f'<div style="padding:4px 0">'
                f'<div class="project-desc">{doc["description"]}</div>'
                f'<div style="margin-bottom:10px">{tags_html}</div>'
                f'<a class="project-link" href="{doc["lien"]}" target="_blank">↗ En savoir plus</a>'
                f'</div>',
                unsafe_allow_html=True
            )


# --- Modèles ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def build_faiss_index(documents):
    model = load_embedding_model()
    descriptions = [doc["description"] for doc in documents]
    embeddings = np.array(model.encode(descriptions)).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model


def retrieve_docs(query, index, model, documents, k=3):
    q = np.array(model.encode([query])).astype("float32")
    _, indices = index.search(q, k)
    return [documents[i] for i in indices[0]]


def generate_response(query, retrieved_docs):
    context = "\n\n".join([
        f"Titre: {doc['title']}\nDescription: {doc['description']}\n"
        f"Tags: {', '.join(doc['tags'])}\nLien: {doc['lien']}"
        for doc in retrieved_docs
    ])
    prompt = f"""Tu es un assistant qui recommande des idées de projets.

Question de l'utilisateur :
{query}

Voici les documents retrouvés :
{context}

Réponds en français. Donne une réponse claire, structurée et utile.
Propose les projets les plus pertinents et explique brièvement pourquoi ils correspondent.
"""
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


main()