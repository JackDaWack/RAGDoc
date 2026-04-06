from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()

class Query(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"Hello": "World"}
