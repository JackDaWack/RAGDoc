import os
from fastapi import FastAPI
from pydantic import BaseModel
import time
import rag

app = FastAPI()

class Query(BaseModel):
    question: str

def prep_rag():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json"))
    embeddings = rag.load_embeddings(path)
    return embeddings


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
