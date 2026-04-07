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
    relevant_docs = rag.retrieve_relevant_documents(query.question, rag.load_documents(rag.path), rag.load_embeddings(os.path.join(rag.path, "embeddings.json")))
    answer = rag.generate_response(query.question, relevant_docs)
    return {"answer": answer}
