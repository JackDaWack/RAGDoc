import os
from fastapi import FastAPI
from pydantic import BaseModel
import time
import rag

app = FastAPI()

class Query(BaseModel):
    question: str

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
def query():
    pass

@app.post("/response")
def get_response():
    pass
