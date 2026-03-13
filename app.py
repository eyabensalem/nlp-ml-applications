# import os
# import re
import streamlit as st
# import plotly.express as px
# import pandas
# import numpy as np
# import spacy
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
# from gensim.models import Word2Vec
# import random
# from scipy.spatial.distance import cosine
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import PCA
# from transformers import pipeline, Mistral3ForConditionalGeneration, MistralCommonBackend, AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# from mistralai import Mistral
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import faiss



pages = {
  "NLP" : [
    st.Page("pages/01_Preprocessing.py", title="Preprocessing"),
    st.Page("pages/02_bow_tfidf.py", title="Bag of Word & TF-IDF"),
    st.Page("pages/03_word2vec.py", title="Word2vec"),
    st.Page("pages/04_rnn_lstm.py", title="RNN LSTM"),
    st.Page("pages/05_encoder_bert.py", title="Encoder BERT"),
    st.Page("pages/06_mistral_chat.py", title="Mistral Chat"),
    st.Page("pages/07_rag.py", title="RAG"),
    st.Page("pages/08_cemantix_game.py", title="Jeu Cemantix"),
    st.Page("pages/09_codenames.py", title="Codenames"),
    #st.Page("pages/10_codenames_spymaster.py", title="Codenames Spymaster"),
  ]
}

pg = st.navigation(pages)
pg.run()
