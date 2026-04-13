import os
import json
import torch
import pdfplumber
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "documents"))

if not os.path.exists(path):
    raise FileNotFoundError(f"Data directory not found: {path}")

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as file:
                documents.append(file.read())
        elif filename.endswith('.pdf'):
            with pdfplumber.open(filepath) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                documents.append(text)
    return documents

def generate_embeddings(documents):
    embeddings = []
    for doc in documents:
        response = client.embeddings.create(input=doc, model="text-embedding-3-small")
        embeddings.append(response.data[0].embedding)
    return embeddings

def retrieve_relevant_documents(query, documents, embeddings, top_k=3):
    query_response = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embedding = query_response.data[0].embedding
    similarities = [torch.nn.functional.cosine_similarity(torch.tensor(query_embedding).unsqueeze(0), torch.tensor(emb).unsqueeze(0)).item() for emb in embeddings]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return [documents[i] for i in top_indices]

def generate_response(query, relevant_docs):
    context = "\n\n".join(relevant_docs)
    messages = [
        {"role": "system", "content": "You are a helpful assistant for clinical document queries."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def store_embeddings(embeddings, filepath):
    with open(filepath, 'w') as f:
        json.dump(embeddings, f)

def load_embeddings(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)