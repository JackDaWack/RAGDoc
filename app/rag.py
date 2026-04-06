import os
import json
import openai
import torch

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                documents.append(file.read())
    return documents

def generate_embeddings(documents):
    embeddings = []
    for doc in documents:
        response = openai.Embedding.create(input=doc, model="text-embedding-3-small")
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

def retrieve_relevant_documents(query, documents, embeddings, top_k=3):
    query_embedding = openai.Embedding.create(input=query, model="text-embedding-3-small")['data'][0]['embedding']
    similarities = [torch.nn.functional.cosine_similarity(torch.tensor(query_embedding).unsqueeze(0), torch.tensor(emb).unsqueeze(0)).item() for emb in embeddings]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return [documents[i] for i in top_indices]

def generate_response(query, relevant_docs):
    context = "\n\n".join(relevant_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()
