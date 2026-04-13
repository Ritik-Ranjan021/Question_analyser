from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
import os
import json
import faiss
import torch
from typing import List, Optional
import importlib.util
import sys
import shutil
import asyncio
import pickle
from pathlib import Path

# Add src folder to path for module imports
_src_path = os.path.dirname(os.path.abspath(__file__))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Import config first
from config import get_config

# Load local rag.py explicitly to avoid name conflict with pip package 'rag'
_rag_path = os.path.join(_src_path, "rag.py")
spec = importlib.util.spec_from_file_location("local_rag", _rag_path)
rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag)

# Load local build_index.py
_build_path = os.path.join(_src_path, "build_index.py")
spec_build = importlib.util.spec_from_file_location("local_build", _build_path)
build_module = importlib.util.module_from_spec(spec_build)
spec_build.loader.exec_module(build_module)

# Import functions
load_index = rag.load_index
retrieve = rag.retrieve
generate_answer = rag.generate_answer
rag_pipeline = rag.rag_pipeline
create_index_if_missing = build_module.create_index_if_missing
gather_chunks = build_module.gather_chunks
build_index = build_module.build_index

# Import utils
from utils import load_text_from_file, chunk_text

# Get configuration
cfg = get_config()

# Global variables for models and indexes
INDEX = None
META = None
EMBED_MODEL = None
Q_INDEX = None
Q_META = []
IS_BUILDING_INDEX = False  # Flag to track if index is being rebuilt


class QueryReq(BaseModel):
    query: str
    top_k: int = 5


class ChatReq(BaseModel):
    query: str
    top_k: int = 5


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup_event():
    """Initialize indexes and models on startup."""
    global INDEX, META, EMBED_MODEL, Q_INDEX, Q_META
    
    # Build the main index if it doesn't exist
    print("Checking for vector database...")
    create_index_if_missing(cfg.data_folder, cfg.index_path, cfg.metadata_path)
    
    # Load the main index
    INDEX, META = load_index(cfg.index_path, cfg.metadata_path)
    
    # Determine device - use configured device if available, otherwise CPU
    device = cfg.embedding_device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        device = "cpu"
    
    EMBED_MODEL = SentenceTransformer(cfg.embedding_model, device=device)
    
    # Attempt to load question-level index and metadata (optional)
    Q_INDEX = None
    Q_META = []
    try:
        if os.path.exists(cfg.question_index_path) and os.path.exists(cfg.question_meta_path):
            Q_INDEX = faiss.read_index(cfg.question_index_path)
            with open(cfg.question_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    Q_META.append(json.loads(line))
            print(f"✓ Loaded question index with {Q_INDEX.ntotal} questions")
    except Exception as e:
        print(f"Could not load question index: {e}")
        Q_INDEX = None
        Q_META = []
    
    print("✓ Server initialized successfully!")


@app.get("/")
def root():
    """Serve the frontpage as the default landing page."""
    return FileResponse("static/frontpage.html", media_type="text/html")


@app.get("/chatpage")
def chatpage_route():
    """Serve the chat page."""
    return FileResponse("static/chatpage.html", media_type="text/html")


@app.get("/api/home")
def home():
    return {"message": "Question Analyzer RAG Backend. Visit / for the frontend."}


@app.post("/api/chat")
def chat(req: ChatReq):
    """
    Chat endpoint: Uses RAG pipeline to answer questions based on vector database.
    """
    if INDEX is None:
        raise HTTPException(status_code=400, detail="Vector database not initialized. Run build_index.py first.")
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty.")
    
    # Ensure top_k is within reasonable limits
    top_k = min(max(1, req.top_k), 20)
    
    # Run the RAG pipeline
    result = rag_pipeline(
        req.query,
        index_path=cfg.index_path,
        meta_path=cfg.metadata_path,
        top_k=top_k
    )
    
    return result


async def rebuild_index_after_upload():
    """Rebuild the index in the background after files are uploaded."""
    global INDEX, META, IS_BUILDING_INDEX
    try:
        print("🔄 Starting index rebuild...")
        items = gather_chunks(cfg.data_folder)
        
        if not items:
            print("⚠️  No data found in data folder")
            IS_BUILDING_INDEX = False
            return False
        
        print(f"📊 Found {len(items)} chunks from documents")
        new_index, embeddings = build_index(items)
        
        if new_index is None:
            print("❌ Failed to build index")
            IS_BUILDING_INDEX = False
            return False
        
        # Create data directory if it doesn't exist
        Path(cfg.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save new index and metadata
        faiss.write_index(new_index, cfg.index_path)
        with open(cfg.metadata_path, "wb") as f:
            pickle.dump(items, f)
        
        # Load and update global variables
        INDEX, META = load_index(cfg.index_path, cfg.metadata_path)
        
        print(f"✅ Index rebuilt successfully with {len(items)} chunks")
        IS_BUILDING_INDEX = False
        return True
        
    except Exception as e:
        print(f"❌ Error rebuilding index: {e}")
        IS_BUILDING_INDEX = False
        return False


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload PDF/TXT files, save to data folder, and rebuild index.
    Returns progress status messages.
    """
    global IS_BUILDING_INDEX
    
    if IS_BUILDING_INDEX:
        raise HTTPException(status_code=409, detail="Index is already being rebuilt. Please wait.")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    # Validate file types
    allowed_extensions = {'.pdf', '.txt'}
    uploaded_files = []
    
    try:
        # Create data folder if it doesn't exist
        data_folder = Path(cfg.data_folder)
        data_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"📤 Uploading {len(files)} file(s)...")
        
        for file in files:
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            # Validate extension
            if file_ext not in allowed_extensions:
                print(f"⚠️  Skipping {file.filename} - unsupported format")
                continue
            
            # Save file to data folder
            file_path = data_folder / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            uploaded_files.append(file.filename)
            print(f"✓ Saved: {file.filename}")
        
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No valid PDF/TXT files provided.")
        
        # Mark that we're building the index
        IS_BUILDING_INDEX = True
        
        # Rebuild index in the background
        await rebuild_index_after_upload()
        
        return {
            "success": True,
            "message": "✅ Files uploaded and index rebuilt successfully!",
            "uploaded_files": uploaded_files,
            "total_files": len(uploaded_files),
            "status": "complete"
        }
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        IS_BUILDING_INDEX = False
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/suggest")
def suggest(req: QueryReq):
    """Legacy endpoint for backward compatibility."""
    if INDEX is None:
        raise HTTPException(status_code=400, detail="Index not found. Build with build_index.py and place in data/")
    retrieved = retrieve(INDEX, META, EMBED_MODEL, req.query, top_k=req.top_k)
    answer = generate_answer(retrieved, req.query)
    return {"retrieved": retrieved, "answer": answer}


class SearchReq(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/questions/search")
def question_search(req: SearchReq):
    """Search for similar questions in the question index."""
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


@app.get("/api/status")
def status():
    """Check if the vector database is initialized."""
    return {
        "vector_db_ready": INDEX is not None,
        "num_chunks": INDEX.ntotal if INDEX is not None else 0,
        "question_index_ready": Q_INDEX is not None,
        "num_questions": len(Q_META) if Q_META else 0,
        "config": {
            "embedding_model": cfg.embedding_model,
            "generation_model": cfg.generation_model,
            "index_path": cfg.index_path,
            "data_folder": cfg.data_folder
        }
    }


@app.get("/api/config")
def get_app_config():
    """Get current application configuration."""
    return {
        "server": {"host": cfg.server_host, "port": cfg.server_port},
        "models": {
            "embedding": cfg.embedding_model,
            "generation": cfg.generation_model,
            "generation_provider": cfg.generation_provider,
            "device": cfg.embedding_device
        },
        "database": {
            "index_path": cfg.index_path,
            "metadata_path": cfg.metadata_path,
            "data_folder": cfg.data_folder
        },
        "rag": {
            "top_k": cfg.rag_top_k,
            "provider": cfg.rag_provider
        }
    }


# Mount static files AFTER all API routes (important for route precedence)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host=cfg.server_host, port=cfg.server_port, reload=False)
