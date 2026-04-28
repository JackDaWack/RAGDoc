import rag
import os
def precompute_embeddings():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "documents"))
    embedding_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json"))
    rag.ingest_data(path, embedding_file)

if __name__ == "__main__":
    precompute_embeddings()