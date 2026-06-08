import os
import json
import torch
import pdfplumber
from openai import OpenAI
import tiktoken

open_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "documents"))

if not os.path.exists(path):
    raise FileNotFoundError(f"Data directory not found: {path}")

#Data Ingestion Functions:
def load_documents():
    pass

def chunk():
    pass

def gen_embeds():
    pass

def store_vectors():
    pass

#Query Handling Functions:
def query_to_embeds():
    pass

def retrieve_candidates():
    pass

#Response Generation:
def build_context():
    pass

def answer_query():
    pass