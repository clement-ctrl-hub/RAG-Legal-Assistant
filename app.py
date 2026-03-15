import streamlit as st
from rag_pipeline import rag_pipeline
from sentence_transformers import SentenceTransformer
import faiss

# charger les modèles une seule fois pour limiter le temps de réflexion
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector_db/code_penal.index")
    return model, index

model, index = load_models()

st.title("⚖️ Assistant juridique togolais")

st.caption(
"⚠️ Cet assistant fournit des informations basées sur le Code pénal togolais et ne remplace pas un avocat."
)

# historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# afficher l'historique
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# input utilisateur
question = st.chat_input("Posez votre question sur le Code pénal togolais")

if question:

    # afficher question
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # spinner pendant la génération
    with st.spinner("Analyse juridique en cours..."):

        answer, docs = rag_pipeline(question)

    # afficher réponse
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # afficher les articles utilisés
    with st.expander("📜 Articles du Code pénal utilisés"):
        for doc in docs[:2]:   # limiter à 2 articles pour la rapidité
            st.write(doc)