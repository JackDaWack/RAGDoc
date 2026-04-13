from fastapi import FastAPI
from pydantic import BaseModel
import time
import rag

app = FastAPI()

class Query(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
def answer_query(query: Query):
    start_time = time.time()
    documents = rag.load_documents(rag.path)
    embeddings = rag.generate_embeddings(documents)
    relevant_docs = rag.retrieve_relevant_documents(query.question, documents, embeddings)
    end_time = time.time()
    print(f"Query processed in {end_time - start_time:.2f} seconds")
    return {"answer": "This is a placeholder answer.", "relevant_docs": relevant_docs}

@app.get("/response")
def get_response():
    return {"response": "This is a placeholder response."}
