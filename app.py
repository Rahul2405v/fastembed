from rag.service import rag_answer
from rag.extract_idx import find as find_ids
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
import uvicorn
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskBody(BaseModel):
    prompt: str
    k: int = 4

@app.post("/ask")
def ask_direct(body: AskBody):
    reply, docs = rag_answer(body.prompt, body.k)
    extract_ids = find_ids(reply)
    return {"answer": reply, "chunks": docs, "extracted_ids": extract_ids}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)