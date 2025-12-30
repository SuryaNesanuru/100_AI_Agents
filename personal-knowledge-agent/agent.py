import os
import json
import numpy as np
import requests
from datetime import date
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

EMBEDDING_MODEL_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
CHAT_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

def read_notes(path="notes.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def embed_texts(texts):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(EMBEDDING_MODEL_URL, headers=headers, json={"inputs": texts})
    response.raise_for_status()
    # Hugging Face returns a list of lists (one embedding per input)
    return response.json()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def store_knowledge(chunks, embeddings):
    records = []
    for text, emb in zip(chunks, embeddings):
        records.append({
            "text": text,
            "embedding": emb,
            "created": date.today().isoformat()
        })
    with open("knowledge.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return records

def load_knowledge():
    with open("knowledge.json", "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve(query, records, top_k=3):
    query_emb = embed_texts([query])[0]
    scored = []
    for r in records:
        score = cosine_similarity(query_emb, r["embedding"])
        scored.append((score, r["text"]))
    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]

def answer_query(query, contexts):
    prompt = f"""
Answer the following question using ONLY the provided notes.\n\nNotes:\n{chr(10).join(contexts)}\n\nQuestion:\n{query}\n"""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    response = requests.post(CHAT_MODEL_URL, headers=headers, json=payload)
    response.raise_for_status()
    # Hugging Face returns a list of dicts with 'generated_text'
    result = response.json()
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    else:
        return str(result)

def main():
    print("Ingesting notes...")
    chunks = read_notes()
    embeddings = embed_texts(chunks)
    records = store_knowledge(chunks, embeddings)

    print("Knowledge base ready.")
    query = input("\nAsk a question: ")
    top_contexts = retrieve(query, records)
    answer = answer_query(query, top_contexts)

    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()
