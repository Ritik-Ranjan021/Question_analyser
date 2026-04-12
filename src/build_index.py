import os
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from utils import load_text_from_file, chunk_text
from config import get_config


def gather_chunks(data_dir):
    """Gather all chunks from text and PDF files in data directory."""
    cfg = get_config()
    items = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return items
    
    # Process all supported files
    for ext in cfg.supported_formats:
        for file_path in data_path.glob(f"*{ext}"):
            print(f"Processing: {file_path}")
            text = load_text_from_file(str(file_path))
            if text:
                chunks = chunk_text(text, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
                for i, c in enumerate(chunks):
                    items.append({
                        "text": c, 
                        "source": str(file_path), 
                        "chunk_id": i
                    })
    
    return items


def build_index(items, model_name=None):
    """Build FAISS index from chunks."""
    cfg = get_config()
    if model_name is None:
        model_name = cfg.embedding_model
    
    if not items:
        print("No items to index")
        return None, None
    
    # Determine device - use configured device if CUDA available, otherwise CPU
    device = cfg.embedding_device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        device = "cpu"
    
    model = SentenceTransformer(model_name, device=device)
    texts = [it["text"] for it in items]
    batch_size = 64
    emb_list = []
    
    print(f"Embedding {len(texts)} chunks...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        emb_list.append(embs)
    
    if emb_list:
        embeddings = np.vstack(emb_list)
    else:
        embeddings = np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")

    # normalize for cosine similarity using inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return index, embeddings


def create_index_if_missing(data_dir=None, index_path=None, meta_path=None):
    """Automatically create index if it doesn't exist."""
    cfg = get_config()
    
    if data_dir is None:
        data_dir = cfg.data_folder
    if index_path is None:
        index_path = cfg.index_path
    if meta_path is None:
        meta_path = cfg.metadata_path
    
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    
    # Check if index already exists
    if index_path.exists() and meta_path.exists():
        print(f"✓ Index already exists at {index_path}")
        return True
    
    print("Building index from scratch...")
    items = gather_chunks(data_dir)
    
    if not items:
        print("No data found in data folder")
        return False

    index, embeddings = build_index(items)
    
    if index is None:
        return False

    # Create data directory if it doesn't exist
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save index and metadata
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(items, f)

    print(f"✓ Index built with {len(items)} chunks")
    print(f"✓ Index saved to {index_path}")
    print(f"✓ Metadata saved to {meta_path}")
    return True


if __name__ == "__main__":
    # Load config and build index
    cfg = get_config()
    create_index_if_missing(cfg.data_folder, cfg.index_path, cfg.metadata_path)
