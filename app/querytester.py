import rag_pipeline as rag
import os

def test_query():
    query = ""
    while query.lower() != "exit":
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        relevant_chunks = rag.build_context(query)
        response = rag.answer_query(relevant_chunks, query)
        print("Response:", response)

if __name__ == "__main__":
    test_query()