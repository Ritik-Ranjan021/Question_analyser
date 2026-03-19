import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils import load_text_from_file, chunk_text


def gather_chunks(data_dir):
    items = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            path = os.path.join(root, fn)
            text = load_text_from_file(path)
            chunks = chunk_text(text)
            for i, c in enumerate(chunks):
                items.append({"text": c, "source": path, "chunk_id": i})
    return items


def build_index(items, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [it["text"] for it in items]
    batch_size = 64
    emb_list = []
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--meta_path", required=True)
    args = parser.parse_args()

    items = gather_chunks(args.data_dir)
    if not items:
        print("No data found in", args.data_dir)
        return

    index, embeddings = build_index(items)

    # save index and metadata
    faiss.write_index(index, args.index_path)
    with open(args.meta_path, "wb") as f:
        pickle.dump(items, f)

    print("Index saved to", args.index_path)


if __name__ == "__main__":
    main()
