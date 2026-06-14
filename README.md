# ⚖️ Assistant Juridique Togolais (RAG)

Assistant juridique basé sur une architecture **Retrieval-Augmented Generation (RAG)** permettant d'interroger le **Code pénal togolais** en langage naturel.

L'application combine la recherche sémantique, les modèles d'embeddings OpenAI et GPT-4.1 Mini afin de fournir des réponses juridiques contextualisées et accompagnées des articles pertinents du Code pénal.

---

## 🚀 Fonctionnalités

* Interrogation du Code pénal togolais en langage naturel
* Recherche sémantique d'articles via FAISS
* Génération de réponses juridiques contextualisées avec GPT-4.1 Mini
* Affichage des articles du Code pénal utilisés pour la réponse
* Affichage d'un score de pertinence pour chaque article récupéré
* Interface conversationnelle développée avec Streamlit

---

## 🛠️ Technologies utilisées

### Intelligence Artificielle & NLP

* OpenAI GPT-4.1 Mini
* OpenAI Embeddings (`text-embedding-3-small`)
* Retrieval-Augmented Generation (RAG)

### Recherche vectorielle

* FAISS

### Développement

* Python
* Streamlit
* NumPy

---

## 🏗️ Architecture

```text
Question utilisateur
        │
        ▼
OpenAI Embeddings
(text-embedding-3-small)
        │
        ▼
FAISS Vector Search
        │
        ▼
Articles pertinents du Code pénal
        │
        ▼
GPT-4.1 Mini
        │
        ▼
Réponse juridique contextualisée
```

---

## 📚 Exemple de réponse

**Question :**

> Quelle est la peine prévue pour le vol ?

**Articles récupérés :**

* Article 415 : Vol simple
* Article 417 : Vol aggravé
* Article 418 : Vol aggravé
* Article 419 : Vol avec violences
* Article 420 : Vol avec mutilation ou invalidité

**Réponse :**

> Selon l'article 415 du Code pénal togolais, le vol simple est puni d'une peine d'emprisonnement d'un (01) à trois (03) ans ainsi que d'une amende de cent mille (100 000) à trois millions (3 000 000) de francs CFA.

---

## 📸 Interface de l'application

![Application](images/app_demo.png)

---

## 🎯 Objectifs du projet

* Construire un système RAG de bout en bout
* Exploiter les embeddings OpenAI pour la recherche juridique
* Développer une interface conversationnelle spécialisée
* Mettre en œuvre une architecture d'IA générative utilisées en entreprise

---

## 🔮 Améliorations futures

* Recherche hybride FAISS + BM25
* API REST avec FastAPI
* Conteneurisation Docker
* Déploiement Cloud (GCP ou Azure)
* Extension à d'autres textes juridiques togolais
* Ajout de jurisprudence et de textes OHADA

```
```
