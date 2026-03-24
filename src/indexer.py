import os
import re
import json
import argparse
import uuid
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils import load_text_from_file


YEAR_RE = re.compile(r"20\d{2}")


def extract_year_from_path(path: str):
    m = YEAR_RE.search(path)
    if m:
        return int(m.group(0))
    return None


def gather_questions(data_dir: str):
    entries = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            path = os.path.join(root, fn)
            text = load_text_from_file(path)
            if not text:
                continue
            qid = str(uuid.uuid4())
            year = extract_year_from_path(path)
            entries.append({
                "id": qid,
                "source": path,
                "year": year,
                "text": text,
            })
    return entries


def build_question_index(entries, model_name="all-MiniLM-L6-v2", batch_size=64):
    model = SentenceTransformer(model_name)
    texts = [e["text"] for e in entries]
    emb_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding questions"):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        emb_list.append(embs)
    if emb_list:
        embeddings = np.vstack(emb_list).astype("float32")
    else:
        embeddings = np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if len(embeddings) > 0:
        index.add(embeddings)
    return index, embeddings


def save_metadata_jsonl(entries, meta_path):
    with open(meta_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build question-level FAISS index and metadata JSONL")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    entries = gather_questions(args.data_dir)
    if not entries:
        print("No questions found in", args.data_dir)
        return

    index, embeddings = build_question_index(entries, model_name=args.model)

    faiss.write_index(index, args.index_path)
    save_metadata_jsonl(entries, args.meta_path)

    print("Saved question index to", args.index_path)
    print("Saved metadata to", args.meta_path)


if __name__ == "__main__":
    main()
