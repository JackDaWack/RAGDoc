import app.rag as rag
import os

def test_query():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json"))
    rag_instance = rag.RAG()
    rag_instance.load_embeddings(path)
    while input("Run query test? (y/n): ").lower() == 'y':
        query = input("Enter your query: ")
        relevant_docs = rag_instance.handle_query(query)
        response = rag_instance.generate_response(query, relevant_docs)
        print("Response:", response)

if __name__ == "__main__":
    test_query()