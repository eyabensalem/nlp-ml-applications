import random
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Configuration de la page
# =========================
st.set_page_config(page_title="Codenames Spymaster", layout="wide")

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 2.3rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 1.2rem;
}
.word-tile {
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    margin-bottom: 10px;
    border: 1px solid #d1d5db;
}
.blue { background-color: #dbeafe; }
.red { background-color: #fee2e2; }
.neutral { background-color: #f3f4f6; }
.black { background-color: #111827; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Codenames Spymaster AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">L’IA connaît les labels et génère un indice pour relier les mots d’une équipe cible.</div>',
    unsafe_allow_html=True
)


# =========================
# Données
# =========================
DEFAULT_WORDS = [
    "chat", "chien", "table", "fusée", "piano",
    "robot", "banque", "forêt", "livre", "laser",
    "montagne", "bouteille", "voiture", "nuage", "tigre",
    "soleil", "neige", "musique", "train", "ordinateur",
    "rivière", "orange", "feu", "miel", "avion"
]

LABELS_TEMPLATE = ["bleu"] * 9 + ["rouge"] * 8 + ["neutre"] * 7 + ["assassin"]


# =========================
# Modèle d'embedding
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


model = load_embedding_model()


# =========================
# Fonctions utilitaires
# =========================
def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_grid() -> pd.DataFrame:
    labels = LABELS_TEMPLATE.copy()
    random.shuffle(labels)
    return pd.DataFrame({
        "word": DEFAULT_WORDS,
        "label": labels
    })


def label_class(label: str) -> str:
    if label == "bleu":
        return "blue"
    if label == "rouge":
        return "red"
    if label == "neutre":
        return "neutral"
    return "black"


def get_embeddings(words: list[str]) -> np.ndarray:
    return model.encode(words)


def cluster_target_words(words: list[str], embeddings: np.ndarray, n_clusters: int) -> pd.DataFrame:
    """
    Regroupe les mots cibles avec Agglomerative Clustering.
    """
    if len(words) <= 1:
        return pd.DataFrame({"word": words, "cluster": [0]})

    n_clusters = min(n_clusters, len(words))
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(embeddings)

    return pd.DataFrame({
        "word": words,
        "cluster": clusters
    })


def score_candidate_clue(clue: str, target_words: list[str], other_words: list[str], assassin_words: list[str]) -> dict:
    """
    Score un indice en fonction de :
    - sa proximité avec les mots cibles
    - sa distance avec les autres mots
    - sa distance avec l'assassin
    """
    all_words = [clue] + target_words + other_words + assassin_words
    embeddings = get_embeddings(all_words)

    clue_vec = embeddings[0].reshape(1, -1)

    target_vecs = embeddings[1:1 + len(target_words)]
    other_start = 1 + len(target_words)
    other_end = other_start + len(other_words)
    other_vecs = embeddings[other_start:other_end]
    assassin_vecs = embeddings[other_end:]

    target_score = cosine_similarity(clue_vec, target_vecs).mean() if len(target_vecs) > 0 else 0
    other_score = cosine_similarity(clue_vec, other_vecs).mean() if len(other_vecs) > 0 else 0
    assassin_score = cosine_similarity(clue_vec, assassin_vecs).mean() if len(assassin_vecs) > 0 else 0

    final_score = target_score - 0.8 * other_score - 1.5 * assassin_score

    return {
        "clue": clue,
        "target_score": float(target_score),
        "other_score": float(other_score),
        "assassin_score": float(assassin_score),
        "final_score": float(final_score)
    }


def build_prompt(cluster_words: list[str], team_name: str) -> str:
    """
    Construit un prompt textuel décrivant la tâche.
    """
    joined_words = ", ".join(cluster_words)
    return (
        f"Tu es le spymaster du jeu Codenames. "
        f"Tu dois aider l'équipe {team_name}.\n"
        f"Propose un seul mot indice en français pour relier ces mots : {joined_words}.\n"
        f"Contraintes :\n"
        f"- un seul mot\n"
        f"- pas de mot identique à la grille\n"
        f"- mot simple, général, clair\n"
        f"- pas de phrase\n"
        f"- éviter les ambiguïtés\n"
    )


def simple_clue_generator(cluster_words: list[str]) -> list[str]:
    """
    Générateur simple d'indices candidats sans LLM.
    On peut remplacer ensuite par un appel Mistral.
    """
    mapping = {
        frozenset(["chat", "chien", "tigre"]): ["animal", "félin", "mammifère"],
        frozenset(["rivière", "neige", "nuage"]): ["eau", "nature", "climat"],
        frozenset(["livre", "musique", "piano"]): ["art", "culture", "création"],
        frozenset(["robot", "ordinateur", "laser"]): ["technologie", "machine", "science"],
        frozenset(["voiture", "train", "avion"]): ["transport", "voyage", "véhicule"],
        frozenset(["banque", "miel", "orange"]): ["commerce", "produit", "marché"],
        frozenset(["forêt", "montagne", "rivière"]): ["nature", "paysage", "terre"],
    }

    cluster_set = frozenset(cluster_words)

    for key, values in mapping.items():
        if cluster_set.issubset(key) or key.issubset(cluster_set):
            return values

    # fallback simple
    return ["nature", "objet", "idée", "groupe", "ensemble"]


# =========================
# Session state
# =========================
if "grid_spymaster" not in st.session_state:
    st.session_state.grid_spymaster = build_grid()

if "spymaster_result" not in st.session_state:
    st.session_state.spymaster_result = None


# =========================
# Sidebar
# =========================
st.sidebar.header("Paramètres")

target_label = st.sidebar.selectbox("Équipe cible", ["bleu", "rouge"])
show_labels = st.sidebar.checkbox("Afficher les labels", value=True)
n_clusters = st.sidebar.slider("Nombre de clusters", 2, 5, 3)

if st.sidebar.button("Nouvelle grille"):
    st.session_state.grid_spymaster = build_grid()
    st.session_state.spymaster_result = None
    st.rerun()


# =========================
# Explication
# =========================
with st.expander("Voir la logique demandée"):
    st.markdown("""
    **Objectif implémenté :**
    - l’IA connaît la **grille labelisée**
    - elle doit **générer un mot indice**
    - pour aider à trouver les mots d’un **label précis**
    - on utilise des **embeddings**
    - puis un **Agglomerative Clustering** sur les vecteurs des mots cibles
    - enfin on choisit l’indice le plus pertinent
    """)


# =========================
# Affichage grille
# =========================
st.subheader("Grille labelisée")

grid_df = st.session_state.grid_spymaster.copy()

for row_idx in range(5):
    cols = st.columns(5)
    for col_idx in range(5):
        idx = row_idx * 5 + col_idx
        row = grid_df.iloc[idx]
        css_class = label_class(row["label"]) if show_labels else "neutral"

        cols[col_idx].markdown(
            f'<div class="word-tile {css_class}">{row["word"]}</div>',
            unsafe_allow_html=True
        )

st.markdown("---")


# =========================
# Génération de l'indice
# =========================
if st.button("Générer un indice pour l’équipe cible", use_container_width=True):
    target_df = grid_df[grid_df["label"] == target_label].copy()
    other_df = grid_df[grid_df["label"] != target_label].copy()
    assassin_df = grid_df[grid_df["label"] == "assassin"].copy()

    target_words = target_df["word"].tolist()
    other_words = other_df["word"].tolist()
    assassin_words = assassin_df["word"].tolist()

    target_embeddings = get_embeddings(target_words)

    clustered_df = cluster_target_words(
        words=target_words,
        embeddings=target_embeddings,
        n_clusters=n_clusters
    )

    cluster_scores = []

    for cluster_id in sorted(clustered_df["cluster"].unique()):
        cluster_words = clustered_df[clustered_df["cluster"] == cluster_id]["word"].tolist()

        if len(cluster_words) < 2:
            continue

        prompt = build_prompt(cluster_words, target_label)
        candidate_clues = simple_clue_generator(cluster_words)

        scored_candidates = []
        for clue in candidate_clues:
            normalized_clue = normalize_text(clue)

            if normalized_clue in [normalize_text(w) for w in grid_df["word"].tolist()]:
                continue

            score_dict = score_candidate_clue(
                clue=normalized_clue,
                target_words=cluster_words,
                other_words=other_words,
                assassin_words=assassin_words
            )
            score_dict["cluster_id"] = int(cluster_id)
            score_dict["cluster_words"] = ", ".join(cluster_words)
            score_dict["prompt"] = prompt
            scored_candidates.append(score_dict)

        if scored_candidates:
            best_cluster_candidate = max(scored_candidates, key=lambda x: x["final_score"])
            cluster_scores.append(best_cluster_candidate)

    if cluster_scores:
        best_global = max(cluster_scores, key=lambda x: x["final_score"])
        st.session_state.spymaster_result = {
            "clusters": cluster_scores,
            "best": best_global
        }
    else:
        st.session_state.spymaster_result = None


# =========================
# Résultats
# =========================
if st.session_state.spymaster_result is not None:
    result = st.session_state.spymaster_result
    best = result["best"]
    clusters_df = pd.DataFrame(result["clusters"]).sort_values(by="final_score", ascending=False)

    st.subheader("Indice recommandé")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mot indice", best["clue"])
    col2.metric("Nombre conseillé", len(best["cluster_words"].split(", ")))
    col3.metric("Score global", f"{best['final_score']:.4f}")

    st.success(
        f"Indice conseillé : **{best['clue']} {len(best['cluster_words'].split(', '))}**"
    )

    st.markdown("### Mots ciblés par cet indice")
    st.write(best["cluster_words"])

    st.markdown("### Prompt construit")
    st.code(best["prompt"])

    st.markdown("### Analyse des clusters")
    display_df = clusters_df[[
        "cluster_id", "clue", "cluster_words",
        "target_score", "other_score", "assassin_score", "final_score"
    ]].copy()

    display_df.columns = [
        "Cluster", "Indice", "Mots du cluster",
        "Score cible", "Score autres", "Score assassin", "Score final"
    ]

    st.dataframe(display_df, use_container_width=True)

else:
    st.info("Clique sur le bouton pour générer un indice pour l’équipe cible.")