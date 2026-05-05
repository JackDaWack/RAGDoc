import os
from pydantic import BaseModel
from fastapi import FastAPI
import time
import rag

app = FastAPI()
class Query(BaseModel):
    question: str

relevant_docs = []
query_string = ""

def prep_rag():
    rag_instance = rag.RAG()
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json"))
    rag_instance.load_embeddings(path)
    return rag_instance


@app.get("/")
def read_root():
    prep_rag()
    return {"Hello": "World"}

@app.post("/query")
def query(data: Query):
    query_string = data.question
    rag_instance = prep_rag()
    relevant_docs = rag_instance.handle_query(data.question)

@app.post("/response")
def get_response():
    rag_instance = prep_rag()
    response = rag_instance.generate_response(query_string, relevant_docs)
    return {"response": response}
