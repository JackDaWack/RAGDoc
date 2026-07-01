import os
import json
from click import prompt
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
    chunk_size = 1000  # Number of tokens per chunk
    for doc in docs:
        tokens = encoding.encode(doc)
        for i in range(0, len(tokens), chunk_size):
            chunk = encoding.decode(tokens[i:i + chunk_size])
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

def retrieve_candidates(query_embedding, top_k=3):
    with open(os.path.join(path, "embeddings.json"), "r") as f:
        stored_embeds = json.load(f)
        candidates = []  
        for chunk, embedding in stored_embeds:
            # Compute similarity (e.g., cosine similarity) between query_embedding and stored_embedding
            similarity = torch.nn.functional.cosine_similarity(torch.tensor(query_embedding), torch.tensor(embedding), dim=0)
            # If it's among the top_k, add the corresponding document chunk to candidates
            if len(candidates) < top_k:
                candidates.append((similarity.item(), chunk))
            else:
                candidates.sort(reverse=True)
                if similarity.item() > candidates[-1][0]:
                    candidates[-1] = (similarity.item(), chunk)
    return candidates
 
#Response Generation:
def build_context(query):
    query_embedding = query_to_embeds(query)
    #should return the top_k most relevant document chunks based on similarity to the query embedding
    candidates = retrieve_candidates(query_embedding)
    context = "\n\n".join([chunk for _, chunk in candidates])
    return context

def answer_query(context, query):
    prompt_info = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = open_ai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt_info}],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()
    