from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
from rag import load_index, retrieve, generate_suggestions


class QueryReq(BaseModel):
    query: str
    top_k: int = 5


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.on_event("startup")
def startup_event():
    global INDEX, META, EMBED_MODEL
    INDEX, META = load_index()
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


@app.get("/")
def root():
    return {"message": "Question analyzer backend. Visit /static for the frontend."}


@app.post("/api/suggest")
def suggest(req: QueryReq):
    if INDEX is None:
        raise HTTPException(status_code=400, detail="Index not found. Build with build_index.py and place in data/")
    retrieved = retrieve(INDEX, META, EMBED_MODEL, req.query, top_k=req.top_k)
    suggestions = generate_suggestions(retrieved, req.query)
    return {"retrieved": retrieved, "suggestions": suggestions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
