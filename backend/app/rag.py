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

class RAG:
    def __init__(self):
        self.embeddings = None
        self.documents = None
        self.ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "documents"))

    def load_embeddings(self, filepath):
        with open(filepath, 'r') as f:
            self.embeddings = json.load(f)

    def load_documents(self, directory):
        self.documents = {}
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                with pdfplumber.open(os.path.join(directory, filename)) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    self.documents[filename] = text
    
    # Ingest data and generate embeddings (run once to create the embedding file)
    def ingest_data(self, directory, embedding_file):
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
            response = self.ai_client.embeddings.create(input=doc["text"], model="text-embedding-3-large")
            embeddings[doc["filename"]] = response.data[0].embedding

        with open(embedding_file, 'w') as f:
            json.dump(embeddings, f)

    def chunk_text(self, text, max_tokens=8191):
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        tokens = encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(encoding.decode(chunk_tokens))
        return chunks

    # Retrieve relevant documents and include their text.
    def handle_query(self, query, top_k=1, threshold=0.3):
        response = self.ai_client.embeddings.create(input=query, model="text-embedding-3-large")
        query_embedding = response.data[0].embedding

        self.load_documents(self.path)
        relevant_docs = []
        for filename, doc_embedding in self.embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(torch.tensor(query_embedding), torch.tensor(doc_embedding), dim=0).item()
            if similarity > threshold:
                #print(f"Document: {filename}")
                relevant_docs.append({
                    "filename": filename,
                    "text": self.documents.get(filename, ""),
                    "similarity": similarity,
                })

        relevant_docs.sort(key=lambda doc: doc["similarity"], reverse=True)
        return relevant_docs[:top_k]

    # Generate a response that is forced to use the retrieved document text.
    def generate_response(self, query, relevant_docs):
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
            f"Documents:\n{context}\n\nQuestion: {query}"
        )
        response = self.ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers only from the provided documents. If there are no documents, say there is no information."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content