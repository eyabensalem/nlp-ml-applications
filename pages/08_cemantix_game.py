import random
import re
import unicodedata

import pandas as pd
import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Jeu Cemantix", layout="wide")

# ══════════════════════════════════════════════
# THÈME — Futuriste Clair (identique à Codenames)
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;600;800;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', system-ui, sans-serif;
    color: #0f172a;
}

/* ── FOND ── */
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

/* ── SCORE BOX ── */
.score-box {
    padding: 18px 16px;
    border-radius: 14px;
    background: linear-gradient(145deg, #f0f9ff, #e0f2fe);
    border: 1.5px solid #7dd3fc;
    text-align: center;
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 800;
    color: #0369a1;
    box-shadow: 0 2px 12px rgba(14,165,233,0.14);
    letter-spacing: 0.04em;
}

/* ── INPUTS ── */
.stTextInput > label,
.stNumberInput > label,
.stSelectbox > label,
.stSlider > label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
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

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;
    font-style: italic;
    font-weight: 400;
}

/* ── SELECTBOX ── */
div[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1.5px solid #c8d8e8 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── SLIDER ── */
div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
}

/* ── PROGRESS BAR ── */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    border-radius: 99px !important;
}
div[data-testid="stProgressBar"] > div {
    background: #e2e8f0 !important;
    border-radius: 99px !important;
    height: 10px !important;
}

/* ── METRICS ── */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #c8d8e8;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
div[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.1rem !important;
    box-shadow: 0 4px 14px rgba(14,165,233,0.28) !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.38) !important;
    filter: brightness(1.06) !important;
}
.stButton > button:disabled {
    background: linear-gradient(135deg, #94a3b8, #cbd5e1) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    opacity: 0.6 !important;
}

/* ── TABS ── */
div[data-testid="stTabs"] > div > div > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.05em !important;
    color: #64748b !important;
    border-radius: 8px 8px 0 0 !important;
}
div[data-testid="stTabs"] > div > div > button[aria-selected="true"] {
    color: #0ea5e9 !important;
    border-bottom: 2px solid #0ea5e9 !important;
}

/* ── DATAFRAME ── */
.stDataFrame {
    border: 1.5px solid #c8d8e8 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── ALERTS ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}

/* success */
div[data-testid="stAlert"][data-baseweb="notification"][kind="positive"],
.stSuccess {
    background: #f0fdf4 !important;
    border: 1.5px solid #86efac !important;
    border-left: 4px solid #22c55e !important;
    color: #14532d !important;
    border-radius: 12px !important;
}
/* warning */
.stWarning {
    background: #fffbeb !important;
    border: 1.5px solid #fcd34d !important;
    border-left: 4px solid #f59e0b !important;
    color: #78350f !important;
    border-radius: 12px !important;
}
/* error */
.stError {
    background: #fef2f2 !important;
    border: 1.5px solid #fca5a5 !important;
    border-left: 4px solid #ef4444 !important;
    color: #7f1d1d !important;
    border-radius: 12px !important;
}
/* info */
.stInfo {
    background: #f0f9ff !important;
    border: 1.5px solid #7dd3fc !important;
    border-left: 4px solid #0ea5e9 !important;
    color: #0c4a6e !important;
    border-radius: 12px !important;
}

/* ── CAPTION ── */
div[data-testid="stCaptionContainer"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-style: italic !important;
    color: #64748b !important;
    font-size: 0.87rem !important;
}

/* ── SIDEBAR ── */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 1px solid #c8d8e8;
}
div[data-testid="stSidebar"] * {
    color: #0f172a !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 {
    font-family: 'Orbitron', monospace !important;
    color: #0ea5e9 !important;
}

/* sidebar inputs */
div[data-testid="stSidebar"] .stTextInput > div > div > input,
div[data-testid="stSidebar"] .stSelectbox > div > div,
div[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: #f8fafc !important;
    border: 1.5px solid #c8d8e8 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
}

/* ── RADIO ── */
div[data-testid="stRadio"] label {
    color: #0f172a !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── CHECKBOX ── */
div[data-testid="stCheckbox"] label {
    color: #0f172a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
}

/* ── HORIZONTAL RULE ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #c8d8e8, transparent);
    margin: 1.2rem 0;
}

/* ── SUBHEADER ── */
h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: #0f172a !important;
    font-size: 1rem !important;
    letter-spacing: 0.08em !important;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: none; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Titre
# ══════════════════════════════════════════════
st.markdown('<div class="main-title">◈ CEMANTIX ◈</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Trouve le mot secret grâce à la similarité sémantique.</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="divider-line"><span class="divider-dot"></span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Chargement des modèles
# ══════════════════════════════════════════════
@st.cache_resource
def load_spacy_model():
    return spacy.load("fr_core_news_sm")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


try:
    nlp = load_spacy_model()
except OSError:
    st.error("Le modèle spaCy 'fr_core_news_sm' n'est pas installé.")
    st.info("Commande : python -m spacy download fr_core_news_sm")
    st.stop()

model = load_embedding_model()


# ══════════════════════════════════════════════
# Données
# ══════════════════════════════════════════════
THEMES = {
    "Animaux":      ["chat", "chien", "lion", "tigre", "oiseau", "poisson", "cheval"],
    "Objets":       ["table", "chaise", "voiture", "ordinateur", "stylo", "livre", "téléphone"],
    "Nature":       ["forêt", "rivière", "montagne", "soleil", "pluie", "fleur", "arbre"],
    "Culture":      ["musique", "cinéma", "poésie", "roman", "peinture", "théâtre", "danse"],
    "Nourriture":   ["fromage", "pain", "pomme", "riz", "gâteau", "pizza", "chocolat"],
    "Technologie":  ["robot", "algorithme", "donnée", "serveur", "application", "réseau", "capteur"],
}


# ══════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════
def remove_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def preprocess_word(word: str) -> str:
    word = word.strip().lower()
    word = remove_accents(word)
    word = re.sub(r"[^\w\s-]", "", word)
    word = re.sub(r"\s+", " ", word).strip()
    doc = nlp(word)
    return " ".join(token.lemma_ for token in doc if not token.is_space)


@st.cache_data
def encode_text(text: str):
    return model.encode([text])


def semantic_similarity(w1: str, w2: str) -> float:
    return float(cosine_similarity(encode_text(w1), encode_text(w2))[0][0])


def similarity_label(score: float) -> str:
    if score >= 0.90:   return "🔥 Brûlant"
    elif score >= 0.75: return "🟢 Très proche"
    elif score >= 0.55: return "🟡 Proche"
    elif score >= 0.35: return "🟠 Moyen"
    else:               return "🔵 Loin"


def score_to_percent(score: float) -> float:
    return round((score + 1) * 50, 2)


def progress_value(pct: float) -> float:
    return min(max(pct / 100, 0.0), 1.0)


def reset_game(secret_word: str, theme_name: str):
    st.session_state.secret_word       = secret_word
    st.session_state.secret_word_clean = preprocess_word(secret_word)
    st.session_state.theme_name        = theme_name
    st.session_state.history           = []
    st.session_state.best_score        = -1.0
    st.session_state.game_won          = False
    st.session_state.game_over         = False


# ══════════════════════════════════════════════
# Session state init
# ══════════════════════════════════════════════
if "secret_word" not in st.session_state:
    default_theme = "Animaux"
    reset_game(random.choice(THEMES[default_theme]), default_theme)


# ══════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════
st.sidebar.markdown("## ◈ Paramètres")

game_mode = st.sidebar.selectbox("Mode de jeu", ["Solo aléatoire", "Solo par thème", "2 joueurs"])
selected_theme = st.sidebar.selectbox("Thème", list(THEMES.keys()))
max_attempts = st.sidebar.slider("Tentatives max", 5, 30, 10)
show_secret = st.sidebar.checkbox("Afficher le mot secret (debug)", value=False)

st.sidebar.markdown("---")

player_secret = ""
if game_mode == "2 joueurs":
    player_secret = st.sidebar.text_input("Joueur 1 : mot secret")

if st.sidebar.button("↻ Nouvelle Partie", use_container_width=True):
    if game_mode == "Solo aléatoire":
        t = random.choice(list(THEMES.keys()))
        reset_game(random.choice(THEMES[t]), t)
    elif game_mode == "Solo par thème":
        reset_game(random.choice(THEMES[selected_theme]), selected_theme)
    elif game_mode == "2 joueurs":
        if not player_secret.strip():
            st.sidebar.warning("Entre un mot secret pour le mode 2 joueurs.")
        else:
            reset_game(player_secret.strip(), "Personnalisé")
    st.rerun()

if show_secret:
    st.sidebar.success(f"Mot secret : {st.session_state.secret_word}")
else:
    st.sidebar.info("Le mot secret est caché.")

st.sidebar.markdown(f"**Thème actuel :** {st.session_state.theme_name}")

if st.sidebar.button("💡 Indice"):
    secret = st.session_state.secret_word
    st.sidebar.info(f"1ère lettre : **{secret[0].upper()}** · {len(secret)} lettres")

if st.sidebar.button("⚑ Abandonner"):
    st.session_state.game_over = True
    st.sidebar.error(f"Le mot secret était : {st.session_state.secret_word}")

st.sidebar.markdown("---")
st.sidebar.caption("Prétraitement : minuscules · accents · ponctuation · lemmatisation")


# ══════════════════════════════════════════════
# Dashboard métriques
# ══════════════════════════════════════════════
status_text = "Gagné 🎉" if st.session_state.game_won else ("Terminé ❌" if st.session_state.game_over else "En cours ▶")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Thème",       st.session_state.theme_name)
m2.metric("Tentatives",  len(st.session_state.history))
m3.metric("Meilleur",    f"{score_to_percent(st.session_state.best_score) if st.session_state.best_score >= 0 else 0}%")
m4.metric("État",        status_text)

pct = score_to_percent(st.session_state.best_score) if st.session_state.best_score >= 0 else 0.0
st.progress(progress_value(pct))
st.caption("Plus le score est élevé, plus tu te rapproches du mot secret.")

st.markdown("---")


# ══════════════════════════════════════════════
# Zone de saisie
# ══════════════════════════════════════════════
can_play = not st.session_state.game_won and not st.session_state.game_over

guess = st.text_input("Propose un mot", placeholder="Exemple : chien", disabled=not can_play)

if st.button("⬡ Tester le mot", use_container_width=True, disabled=not can_play):
    if not guess.strip():
        st.warning("Entre un mot avant de tester.")
    else:
        guess_clean = preprocess_word(guess)
        if not guess_clean:
            st.warning("Le mot est vide après prétraitement.")
        else:
            already_tried = [row["Mot normalisé"] for row in st.session_state.history]
            if guess_clean in already_tried:
                st.warning("Tu as déjà proposé ce mot.")
            else:
                score = semantic_similarity(guess_clean, st.session_state.secret_word_clean)
                score_percent = score_to_percent(score)
                found = guess_clean == st.session_state.secret_word_clean

                if score > st.session_state.best_score:
                    st.session_state.best_score = score

                if found:
                    st.session_state.game_won = True

                result_row = {
                    "Tentative":      len(st.session_state.history) + 1,
                    "Mot saisi":      guess,
                    "Mot normalisé":  guess_clean,
                    "Score cosinus":  round(score, 4),
                    "Score %":        score_percent,
                    "Indice":         "Trouvé 🎉" if found else similarity_label(score),
                }
                st.session_state.history.append(result_row)

                if len(st.session_state.history) >= max_attempts and not st.session_state.game_won:
                    st.session_state.game_over = True

                # ── Résultat ──
                st.markdown("---")
                c1, c2 = st.columns([1, 2])

                with c1:
                    st.markdown(
                        f'<div class="score-box">{score_percent}%</div>',
                        unsafe_allow_html=True
                    )

                with c2:
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid #c8d8e8;border-radius:14px;'
                        f'padding:16px 20px;box-shadow:0 2px 10px rgba(0,0,0,0.05);">'
                        f'<div style="font-size:0.76rem;font-weight:600;color:#94a3b8;'
                        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Mot saisi</div>'
                        f'<div style="font-family:\'Orbitron\',monospace;font-size:1.3rem;'
                        f'font-weight:700;color:#0f172a;">{guess}</div>'
                        f'<div style="font-size:0.80rem;color:#64748b;margin-top:4px;">'
                        f'Normalisé : <code style="background:#f1f5f9;padding:2px 6px;'
                        f'border-radius:4px;color:#0369a1;">{guess_clean}</code></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    if found:
                        st.balloons()
                        st.success(
                            f"🎉 Bravo ! Tu as trouvé **{st.session_state.secret_word}** "
                            f"en **{len(st.session_state.history)}** tentative(s) !"
                        )
                    else:
                        st.info(similarity_label(score))


# ══════════════════════════════════════════════
# Fin de partie
# ══════════════════════════════════════════════
if st.session_state.game_over and not st.session_state.game_won:
    st.error(f"Partie terminée. Le mot secret était : **{st.session_state.secret_word}**")


# ══════════════════════════════════════════════
# Historique
# ══════════════════════════════════════════════
if st.session_state.history:
    st.markdown("---")
    st.subheader("◈ Historique des essais")

    history_df = pd.DataFrame(st.session_state.history)
    tab1, tab2 = st.tabs(["📋 Chronologique", "🏆 Meilleurs essais"])

    with tab1:
        st.dataframe(history_df, use_container_width=True)

    with tab2:
        best_df = history_df.sort_values(by="Score cosinus", ascending=False).reset_index(drop=True)
        best_df.index = best_df.index + 1
        st.dataframe(best_df, use_container_width=True)
        best_row = best_df.iloc[0]
        st.success(
            f"Meilleur essai : **{best_row['Mot saisi']}** — "
            f"{best_row['Score cosinus']} ({best_row['Score %']}%)"
        )
else:
    st.info("⬡ Commence par proposer un mot.")