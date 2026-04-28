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