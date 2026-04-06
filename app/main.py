from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}
