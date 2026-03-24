from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
import os
import json
import faiss
from typing import List
import importlib.util

# Load local rag.py explicitly to avoid name conflict with pip package 'rag'
_rag_path = os.path.join(os.path.dirname(__file__), "rag.py")
spec = importlib.util.spec_from_file_location("local_rag", _rag_path)
rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag)
load_index = rag.load_index
retrieve = rag.retrieve
generate_suggestions = rag.generate_suggestions


QUESTION_INDEX_PATH = "data/question_index.faiss"
QUESTION_META_PATH = "data/questions_meta.jsonl"


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
    # attempt to load question-level index and metadata
    global Q_INDEX, Q_META
    Q_INDEX = None
    Q_META = []
    try:
        if os.path.exists(QUESTION_INDEX_PATH) and os.path.exists(QUESTION_META_PATH):
            Q_INDEX = faiss.read_index(QUESTION_INDEX_PATH)
            with open(QUESTION_META_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    Q_META.append(json.loads(line))
    except Exception:
        Q_INDEX = None
        Q_META = []


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


class SearchReq(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/questions/search")
def question_search(req: SearchReq):
    if Q_INDEX is None or not Q_META:
        raise HTTPException(status_code=400, detail="Question index not found. Build with indexer.py")
    q_emb = EMBED_MODEL.encode([req.query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = Q_INDEX.search(q_emb.astype("float32"), req.top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(Q_META):
            continue
        m = Q_META[idx].copy()
        m.update({"score": float(score)})
        results.append(m)
    return {"results": results}


@app.get("/api/questions/repeats")
def question_repeats(similarity_threshold: float = 0.88, top_k: int = 5):
    """Return pairs of questions with similarity above the threshold."""
    if Q_INDEX is None or not Q_META:
        raise HTTPException(status_code=400, detail="Question index not available")
    # compute embeddings on the fly for all questions
    texts = [m.get("text", "") for m in Q_META]
    emb_model = EMBED_MODEL
    embs = emb_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    tmp_index = faiss.IndexFlatIP(dim)
    tmp_index.add(embs.astype("float32"))

    repeats = []
    for i in range(len(texts)):
        D, I = tmp_index.search(embs[i : i + 1].astype("float32"), top_k + 1)
        for score, j in zip(D[0], I[0]):
            if j == i:
                continue
            if float(score) >= similarity_threshold:
                repeats.append({
                    "q1_id": Q_META[i].get("id"),
                    "q2_id": Q_META[j].get("id"),
                    "q1_year": Q_META[i].get("year"),
                    "q2_year": Q_META[j].get("year"),
                    "score": float(score),
                    "q1_source": Q_META[i].get("source"),
                    "q2_source": Q_META[j].get("source"),
                })
    return {"pairs": repeats}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
