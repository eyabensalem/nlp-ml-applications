import random
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Codenames FR", layout="wide")

FRENCH_WORDS = [
    "chat", "chien", "maison", "voiture", "forêt", "rivière", "soleil", "montagne", "livre", "musique",
    "train", "avion", "banque", "robot", "piano", "laser", "tigre", "neige", "orange", "miel",
    "feu", "ordinateur", "bouteille", "nuage", "table", "fleur", "pain", "fromage", "bateau", "route",
    "lampe", "école", "jardin", "pluie", "orage", "ciel", "lune", "étoile", "boulanger", "chocolat",
    "village", "pont", "fenêtre", "bureau", "chaise", "stylo", "souris", "café", "thé", "valise",
    "hôtel", "plage", "désert", "volcan", "glace", "cascade", "oiseau", "poisson", "cheval", "loup",
    "renard", "pomme", "poire", "citron", "raisin", "marché", "cinéma", "théâtre", "danse", "photo",
    "journal", "radio", "téléphone", "donnée", "algorithme", "réseau", "code", "script", "pipeline",
    "vision", "capteur", "usine", "budget", "contrat", "audit", "client", "projet", "produit", "vente",
    "marge", "bilan", "transport", "science", "nature", "animal", "culture", "énergie", "sport", "temps"
]

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
    max-width: 1380px;
}

.main-title {
    text-align: center;
    font-family: 'Orbitron', monospace;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 0.18em;
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

/* PANELS */
.panel {
    background: #ffffff;
    border: 1px solid #c8d8e8;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 8px 24px rgba(14,165,233,0.07);
    position: relative;
    margin-bottom: 16px;
    overflow: hidden;
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

/* INFO CARDS */
.info-card {
    border-radius: 14px;
    padding: 16px 18px;
    border: 1.5px solid #c8d8e8;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    min-height: 90px;
    position: relative;
    overflow: hidden;
    background: #ffffff;
}

.metric-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.76rem;
    font-weight: 500;
    color: #94a3b8;
    margin-bottom: 6px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.1;
}

/* WORD TILES */
.word-tile {
    border-radius: 12px;
    padding: 0 10px;
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    min-height: 72px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 8px;
    border: 1.5px solid transparent;
    position: relative;
    overflow: hidden;
    cursor: default;
    transition: transform 0.16s cubic-bezier(.34,1.56,.64,1), box-shadow 0.16s ease;
    animation: cardIn 0.35s ease backwards;
}

@keyframes cardIn {
    from { opacity: 0; transform: translateY(14px) scale(0.93); }
    to   { opacity: 1; transform: none; }
}

.word-tile:hover {
    transform: translateY(-2px) scale(1.025);
    z-index: 2;
}

/* glare effect */
.word-tile::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 45%;
    background: linear-gradient(180deg, rgba(255,255,255,0.16), transparent);
    pointer-events: none;
    border-radius: 12px 12px 0 0;
}

/* FORMS */
div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* INPUT LABELS */
.stTextInput > label,
.stNumberInput > label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* INPUTS — fond blanc, texte foncé */
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

/* BUTTONS */
.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 4px 14px rgba(14,165,233,0.28) !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.38) !important;
    filter: brightness(1.06) !important;
}

/* STATUS */
.status-ok {
    padding: 13px 18px;
    border-radius: 12px;
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    border-left: 4px solid #22c55e;
    color: #14532d;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.96rem;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(34,197,94,0.08);
}

.status-warn {
    padding: 13px 18px;
    border-radius: 12px;
    background: #fffbeb;
    border: 1.5px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    color: #78350f;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.96rem;
    font-weight: 500;
}

.status-danger {
    padding: 13px 18px;
    border-radius: 12px;
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
    border-left: 4px solid #ef4444;
    color: #7f1d1d;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.96rem;
    font-weight: 500;
}

/* SIDEBAR */
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

/* CAPTION */
div[data-testid="stCaptionContainer"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-style: italic !important;
    color: #64748b !important;
    font-size: 0.87rem !important;
}

/* RADIO */
div[data-testid="stRadio"] label {
    color: #0f172a !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* DATAFRAME */
.stDataFrame {
    border: 1.5px solid #c8d8e8 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: none; }
}
.panel { animation: fadeUp 0.4s ease backwards; }
</style>
""", unsafe_allow_html=True)

# ── Styles cartes ──
CARD_STYLES = {
    "rouge": {
        "bg": "linear-gradient(145deg,#dc2626,#ef4444)",
        "border": "#fca5a5", "text": "#ffffff",
        "shadow": "0 4px 18px rgba(239,68,68,0.32), inset 0 1px 0 rgba(255,255,255,0.15)",
    },
    "bleu": {
        "bg": "linear-gradient(145deg,#1d4ed8,#3b82f6)",
        "border": "#93c5fd", "text": "#ffffff",
        "shadow": "0 4px 18px rgba(59,130,246,0.32), inset 0 1px 0 rgba(255,255,255,0.15)",
    },
    "neutre": {
        "bg": "linear-gradient(145deg,#64748b,#94a3b8)",
        "border": "#cbd5e1", "text": "#ffffff",
        "shadow": "0 4px 14px rgba(100,116,139,0.22), inset 0 1px 0 rgba(255,255,255,0.12)",
    },
    "assassin": {
        "bg": "linear-gradient(145deg,#0f172a,#1e293b)",
        "border": "#334155", "text": "#94a3b8",
        "shadow": "0 4px 18px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.04)",
    },
    "cache": {
        "bg": "linear-gradient(145deg,#ffffff,#f1f5f9)",
        "border": "#c8d8e8", "text": "#0f172a",
        "shadow": "0 2px 10px rgba(0,0,0,0.07), inset 0 1px 0 rgba(255,255,255,0.9)",
    },
}

TEAM_STYLES = {
    "rouge": {"bg": "linear-gradient(145deg,#dc2626,#ef4444)", "text": "#ffffff", "border": "#fca5a5"},
    "bleu":  {"bg": "linear-gradient(145deg,#1d4ed8,#3b82f6)", "text": "#ffffff", "border": "#93c5fd"},
}


# ── Logique ──
def generate_grid():
    words = random.sample(FRENCH_WORDS, 25)
    labels = ["rouge"]*9 + ["bleu"]*8 + ["neutre"]*7 + ["assassin"]
    random.shuffle(labels)
    return [{"word": w, "label": l, "revealed": False} for w, l in zip(words, labels)]


def init_game():
    st.session_state.grid = generate_grid()
    st.session_state.current_team = "rouge"
    st.session_state.guessed_words = []
    st.session_state.scores = {"rouge": 0, "bleu": 0}
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.current_clue = ("", 0)
    st.session_state.correct_answers_for_clue = 0
    st.session_state.master_submitted_clue = False
    st.session_state.status_message = "Nouvelle partie lancée. Bonne chance !"
    st.session_state.view_mode = "Vue Joueurs"


def ensure_state():
    if "grid" not in st.session_state:
        init_game()


def get_card_by_word(word):
    n = word.strip().lower()
    for c in st.session_state.grid:
        if c["word"].lower() == n:
            return c
    return None


def reveal_card(card):
    if not card["revealed"]:
        card["revealed"] = True
        st.session_state.guessed_words.append(card["word"])


def count_remaining(team):
    return sum(1 for c in st.session_state.grid if c["label"] == team and not c["revealed"])


def switch_turn():
    st.session_state.current_team = "bleu" if st.session_state.current_team == "rouge" else "rouge"
    st.session_state.current_clue = ("", 0)
    st.session_state.correct_answers_for_clue = 0
    st.session_state.master_submitted_clue = False


def check_victory():
    for team, msg in [("rouge", "Rouge"), ("bleu", "Bleu")]:
        if count_remaining(team) == 0:
            st.session_state.game_over = True
            st.session_state.winner = team
            st.session_state.status_message = f"🏆 Équipe {msg} remporte la partie !"
            return True
    return False


def submit_clue(clue_word, clue_number):
    if st.session_state.game_over:
        st.session_state.status_message = "La partie est terminée."
        return
    clue_word = clue_word.strip().lower()
    if not clue_word:
        st.session_state.status_message = "Veuillez saisir un mot d'indice."
        return
    st.session_state.current_clue = (clue_word, int(clue_number))
    st.session_state.correct_answers_for_clue = 0
    st.session_state.master_submitted_clue = True
    st.session_state.status_message = (
        f"⬡ Indice [{st.session_state.current_team}] : « {clue_word} » — {clue_number} mot(s)"
    )


def handle_guess(guess_word):
    if st.session_state.game_over:
        st.session_state.status_message = "La partie est terminée."
        return
    if not st.session_state.master_submitted_clue:
        st.session_state.status_message = "Le maître du jeu doit d'abord soumettre un indice."
        return
    card = get_card_by_word(guess_word)
    if card is None:
        st.session_state.status_message = "Mot introuvable dans la grille."
        return
    if card["revealed"]:
        st.session_state.status_message = "Ce mot a déjà été révélé."
        return
    reveal_card(card)
    t = st.session_state.current_team
    other = "bleu" if t == "rouge" else "rouge"
    if card["label"] == t:
        st.session_state.scores[t] += 1
        st.session_state.correct_answers_for_clue += 1
        st.session_state.status_message = f"✓ « {card['word']} » — correct !"
        if check_victory(): return
        if st.session_state.correct_answers_for_clue >= st.session_state.current_clue[1]:
            st.session_state.status_message += " Maximum atteint — tour suivant."
            switch_turn()
    elif card["label"] == other:
        st.session_state.scores[other] += 1
        st.session_state.status_message = f"✗ « {card['word']} » appartient à l'équipe adverse — tour changé."
        if check_victory(): return
        switch_turn()
    elif card["label"] == "neutre":
        st.session_state.status_message = f"◈ « {card['word']} » est neutre — tour changé."
        switch_turn()
    else:
        st.session_state.game_over = True
        st.session_state.winner = other
        st.session_state.status_message = f"☠ Assassin ! Partie terminée — équipe {other} gagne."


# ── Helpers ──
def render_info_card(title, value, bg, text="#0f172a", border="#c8d8e8"):
    st.markdown(
        f'<div class="info-card" style="background:{bg};color:{text};border-color:{border};">'
        f'<div class="metric-label">{title}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_card(card, show_colors=False, delay_ms=0):
    style = CARD_STYLES[card["label"]] if (card["revealed"] or show_colors) else CARD_STYLES["cache"]
    word = card["word"].upper()
    check = "✓ " if card["revealed"] else ""
    opacity = "0.52" if card["revealed"] else "1"
    st.markdown(
        f'<div class="word-tile" style="background:{style["bg"]};color:{style["text"]};'
        f'border-color:{style["border"]};box-shadow:{style["shadow"]};'
        f'animation-delay:{delay_ms}ms;opacity:{opacity};">'
        f'{check}{word}</div>',
        unsafe_allow_html=True,
    )


def render_status(msg, kind="ok"):
    css = {"ok": "status-ok", "warn": "status-warn", "danger": "status-danger"}.get(kind, "status-ok")
    st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)


# ── Main ──
def main():
    ensure_state()

    st.markdown('<div class="main-title">◈ CODENAMES ◈</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Édition Française — Jeu de déduction et d\'espions</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider-line"><span class="divider-dot"></span></div>', unsafe_allow_html=True)

    current_team = st.session_state.current_team

    # Sidebar
    st.sidebar.markdown("## ◈ Contrôles")
    st.session_state.view_mode = st.sidebar.radio(
        "Mode d'affichage",
        ["Vue Joueurs", "Vue Maître du Jeu"],
        index=0 if st.session_state.view_mode == "Vue Joueurs" else 1,
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Nouvelle Partie", use_container_width=True):
        init_game()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Tour actuel :** {current_team.capitalize()}")
    st.sidebar.markdown(f"**Rouges restantes :** {count_remaining('rouge')}")
    st.sidebar.markdown(f"**Bleues restantes :** {count_remaining('bleu')}")

    # Dashboard
    t1, t2, t3, t4 = st.columns(4)
    ts = TEAM_STYLES[current_team]
    with t1:
        render_info_card("Équipe active", current_team.capitalize(), bg=ts["bg"], text=ts["text"], border=ts["border"])
    cw, cn = st.session_state.current_clue
    with t2:
        render_info_card("Indice en cours", f"{cw} · {cn}" if cw else "—",
                         bg="linear-gradient(145deg,#f0f9ff,#e0f2fe)", text="#0369a1", border="#7dd3fc")
    with t3:
        render_info_card("Score Rouge", st.session_state.scores["rouge"],
                         bg=TEAM_STYLES["rouge"]["bg"], text="#fff", border="#fca5a5")
    with t4:
        render_info_card("Score Bleu", st.session_state.scores["bleu"],
                         bg=TEAM_STYLES["bleu"]["bg"], text="#fff", border="#93c5fd")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        render_info_card("Rouges restantes", count_remaining("rouge"),
                         bg="linear-gradient(145deg,#fff5f5,#fee2e2)", text="#991b1b", border="#fca5a5")
    with r2:
        render_info_card("Bleues restantes", count_remaining("bleu"),
                         bg="linear-gradient(145deg,#eff6ff,#dbeafe)", text="#1e40af", border="#93c5fd")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Status
    if st.session_state.game_over:
        render_status(st.session_state.status_message, "danger" if st.session_state.winner else "warn")
    else:
        render_status(st.session_state.status_message, "ok")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Panels saisie
    lp, rp = st.columns([1.2, 1])

    with lp:
        st.markdown('<div class="panel"><div class="section-title">⬡ Indice du Maître du Jeu</div>', unsafe_allow_html=True)
        with st.form("master_form", clear_on_submit=False):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                clue_input = st.text_input("Mot d'indice", placeholder="ex : animal")
            with c2:
                clue_num_input = st.number_input("Nombre", min_value=1, max_value=9, value=1, step=1)
            with c3:
                st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
                if st.form_submit_button("Valider", use_container_width=True):
                    submit_clue(clue_input, clue_num_input)
                    st.rerun()
        if st.session_state.current_clue[0]:
            st.caption(
                f"Actif : « {st.session_state.current_clue[0]} — {st.session_state.current_clue[1]} »"
                f"  ·  Réussies : {st.session_state.correct_answers_for_clue}/{st.session_state.current_clue[1]}"
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with rp:
        st.markdown('<div class="panel"><div class="section-title">⬡ Réponse des Joueurs</div>', unsafe_allow_html=True)
        with st.form("player_form", clear_on_submit=True):
            g1, g2 = st.columns([4, 1])
            with g1:
                guess_input = st.text_input("Votre mot", placeholder="Tapez un mot de la grille")
            with g2:
                st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
                if st.form_submit_button("Jouer", use_container_width=True):
                    handle_guess(guess_input)
                    st.rerun()
        if st.session_state.guessed_words:
            st.caption("Révélés : " + "  ·  ".join(st.session_state.guessed_words))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Grille
    st.markdown('<div class="panel"><div class="section-title">⬡ Grille de Jeu</div>', unsafe_allow_html=True)
    show_colors = st.session_state.view_mode == "Vue Maître du Jeu"
    for row in range(5):
        cols = st.columns(5)
        for col in range(5):
            idx = row * 5 + col
            with cols[col]:
                render_card(st.session_state.grid[idx], show_colors, delay_ms=idx * 28)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Scores
    st.markdown('<div class="panel"><div class="section-title">⬡ Tableau des Scores</div>', unsafe_allow_html=True)
    score_df = pd.DataFrame({
        "Équipe": ["Rouge 🔴", "Bleu 🔵"],
        "Score": [st.session_state.scores["rouge"], st.session_state.scores["bleu"]],
        "Cartes restantes": [count_remaining("rouge"), count_remaining("bleu")],
    })
    st.dataframe(score_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Légende maître
    if show_colors:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="section-title">⬡ Légende</div>', unsafe_allow_html=True)
        leg = st.columns(4)
        for col, (lbl, name, desc) in zip(leg, [
            ("rouge","ROUGE","Votre équipe"),
            ("bleu","BLEU","Équipe adverse"),
            ("neutre","NEUTRE","Pas de point"),
            ("assassin","ASSASSIN","Fin immédiate"),
        ]):
            with col:
                s = CARD_STYLES[lbl]
                st.markdown(
                    f'<div class="word-tile" style="background:{s["bg"]};color:{s["text"]};'
                    f'border-color:{s["border"]};box-shadow:{s["shadow"]};min-height:58px;font-size:0.78rem;">'
                    f'<div><div style="font-weight:700">{name}</div>'
                    f'<div style="opacity:0.65;font-size:0.70rem;font-weight:400">{desc}</div></div></div>',
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()