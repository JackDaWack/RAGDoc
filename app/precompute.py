import rag_pipeline as rag

def precompute_embeddings():
    docs = rag.load_documents()
    chunks = rag.chunk(docs)
    embeds = rag.gen_embeds(chunks)
    rag.store_vectors(embeds)

if __name__ == "__main__":
    precompute_embeddings()