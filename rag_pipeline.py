import json
import faiss
import requests
from sentence_transformers import SentenceTransformer
# charger les données
with open("data/articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# charger embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# charger index FAISS
index = faiss.read_index("vector_db/code_penal.index")

def retrieve_articles(query, k=1):

    query_embedding = model.encode([query])

    D, I = index.search(query_embedding, k)

    docs = [articles[i] for i in I[0]]

    #docs = [articles[i][0] for i in I[0]]

    return docs


def rag_pipeline(question):

    docs = retrieve_articles(question)

    context = "\n\n".join(docs)

    prompt = f"""
Tu es un assistant juridique spécialisé dans le droit togolais.
Réponds uniquement à partir des articles fournis.
Ta réponse doit commencer par :
"Selon l'article X du Code pénal togolais..."
Règles importantes :
cite uniquement les articles présents dans les sources
Ne cite pas d'autres articles
Si l'information n'est pas dans les articles, dis que tu ne sais pas.

Articles :
{context}

Question :
{question}

Réponse :
"""
#appel Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 120 #200, 300 pour un grand model (plus de contexte)
            }
        }
    )

    answer = response.json()["response"]
    return answer, docs