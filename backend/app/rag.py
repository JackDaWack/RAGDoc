import os
import json
import torch
import pdfplumber
from openai import OpenAI
import tiktoken

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
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    max_tokens = 8191

    for doc in documents:
        # Truncate text if it exceeds the maximum token limit
        tokens = encoding.encode(doc["text"])
        if len(tokens) > max_tokens:
            doc["text"] = encoding.decode(tokens[:max_tokens])
        response = client.embeddings.create(input=doc["text"], model="text-embedding-3-large")
        embeddings[doc["filename"]] = response.data[0].embedding
    # Save embeddings.
    with open(embedding_file, 'w') as f:
        json.dump(embeddings, f)

def handle_query(query, embeddings):
    # Generate embedding for the query.
    response = client.embeddings.create(input=query, model="text-embedding-3-large")
    query_embedding = response.data[0].embedding
    # Retrieve relevant documents based on relevance to query.
    relevant_docs = []
    for filename, doc_embedding in embeddings.items():
        # Calculate cosine similarity (or any other similarity metric).
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(query_embedding), torch.tensor(doc_embedding), dim=0).item()
        if similarity > 0.5:  # Threshold for relevance
            relevant_docs.append(filename)
    return relevant_docs

def generate_response(query, relevant_docs):
    # Generate a response using the relevant documents.
    context = "\n".join([f"Document: {doc}" for doc in relevant_docs])
    prompt = f"Answer the following question based on the provided documents:\n\n{context}\n\nQuestion: {query}"
    
    encoding = tiktoken.encoding_for_model("gpt-4o")
    token_count = len(encoding.encode(prompt))

    if token_count > 8191:
        available_tokens = 8191 - len(encoding.encode(f"Answer the following question based on the provided documents:\n\nQuestion: {query}"))
        truncated_context = encoding.decode(encoding.encode(context)[:available_tokens])
        prompt = f"Answer the following question based on the provided documents:\n\n{truncated_context}\n\nQuestion: {query}"
    
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}])
    return response.choices[0].message.content