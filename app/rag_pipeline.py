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
        embeds.append((chunk, response.data[0].embedding))
    return embeds

def store_vectors(embeds=None):
    if embeds is None:
        embeds = gen_embeds()
    else:
        with open(os.path.join(path, "embeddings.json"), "w") as f:
            json.dump(embeds, f)

#Query Handling Functions:
def query_to_embeds(query):
    response = open_ai.embeddings.create(input=query, model="text-embedding-3-large")
    return response.data[0].embedding

def retrieve_candidates(query_embedding, top_k=5):
    with open(os.path.join(path, "embeddings.json"), "r") as f:
        stored_embeds = json.load(f)
        candidates = []  
    for i, stored_embedding in enumerate(stored_embeds):
        # Compute similarity (e.g., cosine similarity) between query_embedding and stored_embedding
        # If it's among the top_k, add the corresponding document chunk to candidates
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(query_embedding), torch.tensor(stored_embedding), dim=0)
        if len(candidates) < top_k:
            candidates.append((similarity, i))
        else:
            candidates.sort(reverse=True)
            if similarity > candidates[-1][0]:
                candidates[-1] = (similarity, i)
    return candidates
 
#Response Generation:
def build_context(query):
    query_embedding = query_to_embeds(query)
    candidates = retrieve_candidates(query_embedding)
    return candidates

def answer_query(embeds, query):
    context = " ".join([chunk for _, chunk in embeds])
    prompt = f"Answer the following query based on the context provided:\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
    response = open_ai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()
    