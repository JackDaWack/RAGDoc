import app.rag as rag
import os
def precompute_embeddings():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "documents"))
    embedding_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json"))
    rag_instance = rag.RAG()
    rag_instance.ingest_data(path, embedding_file)

if __name__ == "__main__":
    precompute_embeddings()