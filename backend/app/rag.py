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

def load_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            with pdfplumber.open(os.path.join(directory, filename)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                documents[filename] = text
    return documents

# Ingest data and generate embeddings (run once to create the embedding file)
def ingest_data(directory, embedding_file):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            with pdfplumber.open(os.path.join(directory, filename)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                documents.append({"filename": filename, "text": text})
    
    embeddings = {}
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    max_tokens = 8191

    for doc in documents:
        tokens = encoding.encode(doc["text"])
        if len(tokens) > max_tokens:
            doc["text"] = encoding.decode(tokens[:max_tokens])
        response = client.embeddings.create(input=doc["text"], model="text-embedding-3-large")
        embeddings[doc["filename"]] = response.data[0].embedding

    with open(embedding_file, 'w') as f:
        json.dump(embeddings, f)

# Retrieve relevant documents and include their text.
def handle_query(query, embeddings, top_k=3, threshold=0.3):
    response = client.embeddings.create(input=query, model="text-embedding-3-large")
    query_embedding = response.data[0].embedding

    documents = load_documents(path)
    relevant_docs = []
    for filename, doc_embedding in embeddings.items():
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(query_embedding), torch.tensor(doc_embedding), dim=0).item()
        print(f"Document: {filename}, Similarity: {similarity:.4f}")
        if similarity > threshold:
            relevant_docs.append({
                "filename": filename,
                "text": documents.get(filename, ""),
                "similarity": similarity,
            })

    relevant_docs.sort(key=lambda doc: doc["similarity"], reverse=True)
    return relevant_docs[:top_k]

# Generate a response that is forced to use the retrieved document text.
def generate_response(query, relevant_docs):
    if not relevant_docs:
        return "I'm so sorry, but I don't have any information to help with that."

    context = "\n\n".join([
        f"Document: {doc['filename']}\n\n{doc['text']}"
        for doc in relevant_docs
    ])

    prompt = (
        "Answer only from the provided documents below. "
        "Do not use any external knowledge beyond these documents. "
        "Format your answer as a concise summary of the relevant information. "
        "If the answer is not in the documents, reply:\n\n"
        '"I don\'t know based on the provided documents."\n\n'
        f"Documents:\n{context}\n\nQuestion: {query}"
    )
    print(f"Context:\n{context}")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers only from the provided documents. If there are no documents, say there is no information."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content