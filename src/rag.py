import os
from pathlib import Path
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from groq import Groq
from config import get_config


def load_index(index_path=None, meta_path=None):
    """Load FAISS index and metadata."""
    cfg = get_config()
    
    if index_path is None:
        index_path = cfg.index_path
    if meta_path is None:
        meta_path = cfg.metadata_path
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print(f"Warning: Index files not found at {index_path} or {meta_path}")
        return None, None
    
    try:
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        print(f"✓ Loaded index with {index.ntotal} vectors")
        return index, meta
    except Exception as e:
        print(f"Error loading index: {e}")
        return None, None


def retrieve(index, meta, embed_model, query, top_k=None):
    """Retrieve top-k relevant chunks from the index using semantic search."""
    cfg = get_config()
    if top_k is None:
        top_k = cfg.rag_top_k
    
    if index is None or meta is None:
        return []
    
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), min(top_k, index.ntotal))
    
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(meta):
            continue
        results.append(meta[idx]["text"])
    
    return results


def generate_answer(retrieved_context, user_query):
    """
    Generate an answer to the user's query based on retrieved context using Groq API.
    """
    cfg = get_config()
    
    # Build the prompt
    context_str = "\n\n".join(retrieved_context) if retrieved_context else "No relevant information found."
    
    prompt = (
        f"Based on the following context, answer the user's question. "
        f"If the answer is not in the context, say 'Information not found in the documents.'\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )

    # Get Groq API key
    groq_api_key = cfg.groq_api_key
    
    if not groq_api_key:
        return "Error: Groq API key not found. Set GROQ_API_KEY environment variable or create groq_key.txt file."
    
    try:
        client = Groq(api_key=groq_api_key)
        
        message = client.chat.completions.create(
            model=cfg.generation_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful study assistant. Answer questions based on the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=cfg.generation_max_tokens,
            temperature=cfg.generation_temperature
        )
        
        answer = message.choices[0].message.content.strip()
        return answer
        
    except Exception as e:
        print(f"Groq API error: {e}")
        return f"Error generating answer: {str(e)}"


def rag_pipeline(query, index_path=None, meta_path=None, top_k=None):
    """
    Complete RAG pipeline: Load index -> Retrieve -> Generate answer using Groq
    """
    cfg = get_config()
    
    if index_path is None:
        index_path = cfg.index_path
    if meta_path is None:
        meta_path = cfg.metadata_path
    if top_k is None:
        top_k = cfg.rag_top_k
    
    # Determine device - use configured device if CUDA available, otherwise CPU
    device = cfg.embedding_device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        device = "cpu"
    
    # Load embedding model
    embed_model = SentenceTransformer(cfg.embedding_model, device=device)
    
    # Load the index
    index, meta = load_index(index_path, meta_path)
    if index is None:
        return {
            "query": query,
            "retrieved_context": [],
            "answer": "Error: Vector database not found. Please run build_index.py first.",
            "success": False
        }
    
    # Retrieve relevant documents
    retrieved_context = retrieve(index, meta, embed_model, query, top_k=top_k)
    
    # Generate answer
    answer = generate_answer(retrieved_context, query)
    
    return {
        "query": query,
        "retrieved_context": retrieved_context,
        "answer": answer,
        "success": True,
        "num_chunks": len(retrieved_context)
    }
