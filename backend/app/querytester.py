import rag
import os

def test_query():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json"))
    embeddings = rag.load_embeddings(path)
    while input("Run query test? (y/n): ").lower() == 'y':
        query = input("Enter your query: ")
        relevant_docs = rag.handle_query(query, embeddings)
        response = rag.generate_response(query, relevant_docs)
        print("Response:", response)

if __name__ == "__main__":
    test_query()