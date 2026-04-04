import os
import json
import openai

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