# 🧠 NLP Playground — Applications interactives en Streamlit

## 📌 Présentation

**NLP Playground** est un projet interactif développé avec **Streamlit** qui regroupe plusieurs démonstrations pratiques autour du **traitement automatique du langage naturel (NLP)**, du **machine learning**, du **deep learning**, des **Transformers** et de la **recherche sémantique**.

L’objectif du projet est de proposer une **suite d’applications pédagogiques et visuelles** permettant de comprendre, tester et comparer différentes techniques de NLP, depuis le prétraitement du texte jusqu’aux modèles modernes comme **BERT**, **Sentence Transformers**, **RAG**, ou encore des jeux sémantiques interactifs.

Ce projet mélange à la fois :

* des **bases théoriques du NLP**
* des **interfaces interactives modernes**
* des **modèles classiques et avancés**
* des **cas d’usage ludiques et pédagogiques**

---

## 🎯 Objectifs du projet

Ce projet a été conçu pour :

* comprendre les étapes clés d’un pipeline NLP
* manipuler différentes techniques de vectorisation de texte
* visualiser des représentations de mots avec Word2Vec
* comparer des architectures séquentielles comme **RNN** et **LSTM**
* explorer des usages concrets des **Transformers**
* intégrer un **LLM externe** via l’API Mistral
* construire un mini système **RAG**
* développer des interfaces interactives de jeux sémantiques comme **Cemantix** et **Codenames**

---

## 🧩 Fonctionnalités du projet

Le projet regroupe plusieurs mini-applications Streamlit.

### 1. Prétraitement NLP

Application dédiée au pipeline de préparation des données textuelles :

* nettoyage du texte
* suppression des balises HTML
* mise en minuscules
* suppression des caractères spéciaux
* tokenisation en mots et en phrases
* stemming avec **NLTK**
* lemmatisation avec **spaCy**
* suppression des **stop words**

👉 Cette partie permet de visualiser étape par étape comment transformer un texte brut en données exploitables pour un modèle NLP.

---

### 2. Vectorisation de texte : BoW, TF et TF-IDF

Application permettant de comparer trois représentations classiques du texte :

* **Bag of Words (BoW)**
* **Term Frequency (TF)**
* **TF-IDF**

Fonctionnalités :

* saisie libre de plusieurs documents
* création automatique des matrices de représentation
* visualisation des poids par terme
* résumé du vocabulaire et des occurrences

👉 Cette partie montre comment convertir du texte en vecteurs numériques exploitables par des algorithmes de machine learning.

---

### 3. Word2Vec : CBOW vs Skip-Gram

Application interactive pour entraîner et comparer deux variantes de **Word2Vec** :

* **CBOW**
* **Skip-Gram**

Fonctionnalités :

* génération d’un corpus artificiel
* entraînement des embeddings
* projection des vecteurs en 3D avec **PCA**
* visualisation interactive avec **Plotly**
* calcul de similarité cosinus entre deux mots
* affichage des mots les plus proches sémantiquement

👉 Cette partie permet de comprendre comment les mots peuvent être représentés dans un espace vectoriel et comment capturer leurs relations sémantiques.

---

### 4. RNN & LSTM pour la prédiction du mot suivant

Application de deep learning basée sur **PyTorch** pour illustrer les modèles séquentiels.

Fonctionnalités :

* construction automatique du vocabulaire
* création des tenseurs d’entraînement
* choix entre **RNN** et **LSTM**
* réglage des hyperparamètres
* entraînement du modèle
* visualisation pas à pas de l’inférence
* affichage des états cachés et, pour le LSTM, de l’état mémoire

👉 Cette partie sert à comprendre comment un modèle séquentiel traite une phrase et prédit le mot suivant.

---

### 5. BERT : analyse de sentiment et question-réponse

Application basée sur les modèles **Transformers** de Hugging Face.

Fonctionnalités :

* analyse de sentiment avec un modèle DistilBERT
* question-réponse extractive à partir d’un contexte
* affichage de la réponse et du score de confiance

👉 Cette partie montre deux usages concrets de BERT :

* la **classification de texte**
* l’**extraction d’information**

---

### 6. Chat avec Mistral

Application connectée à l’API **Mistral** permettant plusieurs tâches :

* chat / question-réponse
* résumé automatique
* génération de code Python

Fonctionnalités :

* configuration via fichier `.env`
* appel API en temps réel
* adaptation du prompt selon la tâche choisie

👉 Cette partie montre comment intégrer un **LLM externe** dans une application NLP.

---

### 7. Mini système RAG — Recherche de projets

Application de **Retrieval-Augmented Generation** combinant :

* **Sentence Transformers** pour les embeddings
* **FAISS** pour la recherche vectorielle
* **Mistral** pour la génération de réponse

Fonctionnalités :

* indexation de documents/projets
* recherche sémantique à partir d’une requête
* récupération des documents les plus pertinents
* génération d’une réponse structurée à partir du contexte retrouvé

👉 Cette partie illustre un pipeline RAG simple mais complet.

---

### 8. Jeu Cemantix

Jeu inspiré de **Cemantix**, basé sur la similarité sémantique entre mots.

Fonctionnalités :

* mot secret à deviner
* calcul de similarité cosinus entre le mot proposé et le mot cible
* normalisation du texte :

  * minuscules
  * suppression des accents
  * suppression de la ponctuation
  * lemmatisation
* score de proximité sémantique
* mode solo, thème, ou 2 joueurs
* historique des tentatives

👉 Cette partie transforme les embeddings en expérience ludique.

---

### 9. Jeu Codenames en français

Version simplifiée du jeu **Codenames** en français avec interface Streamlit.

Fonctionnalités :

* génération d’une grille de 25 mots
* gestion des équipes rouge et bleue
* vue joueur / vue maître du jeu
* système d’indices
* suivi du score
* révélation des cartes
* gestion de la carte assassin

👉 Cette partie montre comment combiner logique de jeu, UI et gestion d’état avec Streamlit.

---

## 🛠️ Technologies utilisées

### Langage

* Python

### Framework

* Streamlit

### NLP / ML / DL

* spaCy
* NLTK
* scikit-learn
* gensim
* PyTorch
* transformers
* sentence-transformers
* FAISS

### Visualisation

* Plotly
* Pandas

### LLM / API

* Mistral API
* python-dotenv

---

## 📂 Structure suggérée du projet

Tu peux organiser ton dépôt comme ceci :

```bash
nlp-playground/
│
├── 01_Preprocessing.py
├── 02_bow_tfidf.py
├── 03_word2vec.py
├── 04_rnn_lstm.py
├── 05_encoder_bert.py
├── 06_mistral_chat.py
├── 07_rag.py
├── 08_cemantix.py
├── 09_codenames.py
│
├── requirements.txt
├── .env.example
├── README.md
└── assets/
```

Si tu veux une version plus propre pour GitHub, tu peux aussi mettre chaque application dans un dossier `pages/` pour créer une vraie **multi-page app Streamlit**.

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/eyabensalem/NLP-Playground-
cd ton-repo
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l’environnement

Sous Windows :

```bash
venv\Scripts\activate
```

Sous Mac/Linux :

```bash
source venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 📦 Dépendances principales

Exemple de `requirements.txt` :

```txt
streamlit
pandas
numpy
plotly
scikit-learn
gensim
torch
transformers
sentence-transformers
faiss-cpu
spacy
nltk
python-dotenv
mistralai
```

---

## 🧠 Ressources supplémentaires à installer

### Télécharger le modèle spaCy français

```bash
python -m spacy download fr_core_news_sm
```

### Télécharger les ressources NLTK

Les ressources NLTK sont téléchargées automatiquement dans l’application, mais tu peux aussi les installer manuellement :

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

---

## 🔐 Configuration de l’API Mistral

Crée un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=ta_cle_api_ici
```

Exemple de fichier `.env.example` :

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Lancer une application

Exemple pour lancer une page Streamlit :

```bash
streamlit run app_word2vec.py
```

Ou :

```bash
streamlit run app_cemantix.py
```

Si tu organises le projet en app multi-pages :

```bash
streamlit run app.py
```

---
---

# 📸 Captures de l'application

## Page 1 — Prétraitement NLP

![Préprocessing](screenshots/p1_1.png)
![Préprocessing](screenshots/p1_2.png)
![Préprocessing](screenshots/p1_3.png)
![Préprocessing](screenshots/p1_4.png)

Interface permettant de visualiser les différentes étapes du pipeline NLP :
nettoyage du texte, tokenisation, lemmatisation et suppression des stop words.

---

## Page 2 — Vectorisation du texte (BoW / TF / TF-IDF)

![Vectorisation](screenshots/p2_1.png)
![Vectorisation](screenshots/p2_2.png)
![Vectorisation](screenshots/p2_3.png)
![Vectorisation](screenshots/p2_4.png)

Comparaison entre différentes méthodes de vectorisation du texte.

---

## Page 3 — Word2Vec

![Word2Vec](screenshots/p3_1.png)
![Word2Vec](screenshots/p3_2.png)
![Word2Vec](screenshots/p3_3.png)

Visualisation des embeddings de mots et calcul de similarité sémantique.

---

## Page 4 — RNN / LSTM

![RNN](screenshots/p4_1.png)
![RNN](screenshots/p4_2.png)
![RNN](screenshots/p4_3.png)

Modèles séquentiels pour la prédiction du mot suivant.

---

## Page 5 — BERT

![BERT](screenshots/p5_1.png)
![BERT](screenshots/p5_2.png)

Analyse de sentiment et question-réponse avec Transformers.

---

## Page 6 — Chat avec Mistral

![Mistral](screenshots/p6_1.png)
![Mistral](screenshots/p6_2.png)

Interface de chat permettant d'interagir avec un LLM.

---

## Page 7 — RAG

![RAG](screenshots/p7_1.png)

Recherche sémantique avec Sentence Transformers et FAISS.

---

## Page 8 — Jeu Cemantix

![Cemantix](screenshots/p8_1.png)
![Cemantix](screenshots/p8_2.png)

Jeu basé sur la similarité sémantique pour deviner un mot secret.

---

## Page 9 — Jeu Codenames

![Codenames](screenshots/p9_1.png)
![Codenames](screenshots/p9_2.png)
![Codenames](screenshots/p9_3.png)
![Codenames](screenshots/p9_4.png)

Version interactive du jeu Codenames avec gestion des équipes et des indices.

---

## 📚 Concepts NLP couverts

Ce projet permet de travailler les notions suivantes :

* nettoyage de texte
* tokenisation
* stemming
* lemmatisation
* stop words
* Bag of Words
* TF
* TF-IDF
* embeddings de mots
* similarité cosinus
* Word2Vec
* réduction de dimension avec PCA
* RNN
* LSTM
* Transformers
* BERT
* analyse de sentiment
* question-réponse extractive
* embeddings de phrases
* recherche vectorielle
* RAG
* interaction avec un LLM
* gamification du NLP

---

## 🚀 Améliorations possibles

Quelques pistes d’évolution :

* transformer le projet en vraie application **multi-pages**
* ajouter une page d’accueil avec navigation
* intégrer des datasets réels au lieu de données simulées
* ajouter une persistance des scores pour les jeux
* proposer davantage de modèles NLP pré-entraînés
* déployer l’application sur **Streamlit Community Cloud**
* ajouter des tests unitaires
* dockeriser le projet


---

## 👤 Auteur

**Eya Ben Salem**
Data Analytics / NLP & AI Projects

