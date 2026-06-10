import os
import json
import torch
import pdfplumber
from openai import OpenAI
import tiktoken

open_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "documents"))

if not os.path.exists(path):
    raise FileNotFoundError(f"Data directory not found: {path}")

#Data Ingestion Functions:
def load_documents(filepath=path):
    docs = []
    for filename in os.listdir(filepath):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(filepath, filename)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages)
                docs.append(text)
    return docs

def chunk(docs=None):
    if docs is None:
        docs = load_documents()
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    max_tokens = 8191
    chunks = []
    for doc in docs:
        tokens = encoding.encode(doc)
        for i in range(0, len(tokens), max_tokens):
            chunk = encoding.decode(tokens[i:i + max_tokens])
            chunks.append(chunk)
    return chunks

def gen_embeds(chunks=None):
    if chunks is None:
        chunks = chunk()
    embeds = []
    for chunk in chunks:
        response = open_ai.embeddings.create(input=chunk, model="text-embedding-3-large")
        embeds.append(response.data[0].embedding)
    return embeds

def store_vectors(embeds=None):
    if embeds is None:
        embeds = gen_embeds()
    else:
        with open(os.path.join(path, "embeddings.json"), "w") as f:
            json.dump(embeds, f)

#Query Handling Functions:
def query_to_embeds():
    pass

def retrieve_candidates():
    pass

#Response Generation:
def build_context():
    pass

def answer_query():
    pass