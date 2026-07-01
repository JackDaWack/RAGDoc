import rag_pipeline as rag
import os

def test_query():
    query = ""
    while query.lower() != "exit":
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        relevant_chunks = rag.build_context(query)
        print(f"Relevant Chunks for the query '{query}': {relevant_chunks}")

if __name__ == "__main__":
    test_query()