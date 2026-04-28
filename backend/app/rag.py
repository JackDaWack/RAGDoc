import os
import json
import torch
import pdfplumber
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "documents"))

if not os.path.exists(path):
    raise FileNotFoundError(f"Data directory not found: {path}")

def load_embeddings(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

#Ingest data and generate embeddings (run once to create the embedding file)
def ingest_data(directory, embedding_file):
    #Document loading.
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            with pdfplumber.open(os.path.join(directory, filename)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages)
                documents.append({"filename": filename, "text": text})
    #Embedding generation.
    embeddings = {}
    for doc in documents:
        response = client.embeddings.create(input=doc["text"], model="text-embedding-3-small")
        embeddings[doc["filename"]] = response.data[0].embedding
    # Save embeddings.
    with open(embedding_file, 'w') as f:
        json.dump(embeddings, f)

def handle_query(query, documents, embeddings):
    # Generate embedding for the query.
    response = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embedding = response.data[0].embedding
    # Retrieve relevant documents based on cosine similarity.
    relevant_docs = []
    for doc in documents:
        doc_embedding = embeddings.get(doc["filename"])
        if doc_embedding:
            similarity = cosine_similarity(query_embedding, doc_embedding)
            if similarity > 0.7:  # Threshold for relevance
                relevant_docs.append(doc)
    return relevant_docs