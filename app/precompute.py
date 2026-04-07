import rag
import os
def precompute_embeddings():

    docs = rag.load_documents(rag.path)
    embeddings = rag.generate_embeddings(docs)
    embeddings_path = os.path.join(os.path.dirname(rag.path), "embeddings.json")
    rag.store_embeddings(embeddings, embeddings_path)

if __name__ == "__main__":
    precompute_embeddings()