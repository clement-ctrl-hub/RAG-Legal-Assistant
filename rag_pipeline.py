# Chargement des clés API
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


import json
import faiss
# charger les données
with open("data/articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# charger index FAISS de l'embedding et du score de similarité
index = faiss.read_index("vector_db/code_penal.index")

def retrieve_articles(query, k=8):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_embedding = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    D, I = index.search(query_embedding, k)

    results = []

    for idx, distance in zip(I[0], D[0]):

        score = round(
            (1 / (1 + float(distance))) * 100,
            1
        )

        results.append({
            "article": articles[idx],
            "distance": float(distance),
            "score": score
        })

    return results


def rag_pipeline(question):

    docs = retrieve_articles(question)

    context = "\n\n".join([doc["article"] for doc in docs])

    prompt = f"""
Tu es un assistant juridique spécialisé dans le droit togolais.

Règles :
- Réponds uniquement à partir des articles fournis
- Si la question est générale et les articles sont spécifiques, précise-le
- Ne généralise jamais un cas particulier
- Si tu n'es pas sûr, dis que la réponse dépend du cas

Articles du code pénal togolais :
{context}

Question :
{question}

Réponse :
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un juriste spécialisé dans le droit togolais."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"⚠️ Erreur OpenAI : {str(e)}"

    return answer, docs