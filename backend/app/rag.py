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
    max_tokens = 8000  # Safety margin below 8192 limit
    
    for doc in documents:
        # Split document into chunks if it's too long
        # Rough estimate: 1 token ≈ 4 characters
        words = doc.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_tokens * 4:  # Convert token limit to characters
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Generate embeddings for each chunk and average them
        chunk_embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(input=chunk, model="text-embedding-3-small")
            chunk_embeddings.append(response.data[0].embedding)
        
        # Average embeddings across chunks
        if chunk_embeddings:
            avg_embedding = [sum(x) / len(chunk_embeddings) for x in zip(*chunk_embeddings)]
            embeddings.append(avg_embedding)
    
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