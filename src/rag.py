import os
from pathlib import Path
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests


def load_index(index_path="data/index.faiss", meta_path="data/metadata.pkl"):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def retrieve(index, meta, embed_model, query, top_k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(meta):
            continue
        results.append(meta[idx]["text"])
    return results


def generate_suggestions(retrieved_texts, user_query, model_name="google/flan-t5-base"):
    prompt = (
        "You are an assistant that analyzes past exam questions and returns a concise study plan.\n\n"
        "Context (excerpts):\n"
        + "-----\n"
        + "\n\n".join(retrieved_texts)
        + "\n-----\n\n"
        "Instructions:\n"
        "- From the context, list the top 5 important concepts to learn (one per line).\n"
        "- Provide 3 practice questions inspired by the excerpts (numbered).\n"
        "- Suggest 4 short, high-quality resources (title + short link or description).\n"
        "- Be concise and use plain text with clear section headers: Concepts, Practice Questions, Resources.\n\n"
        "User request: "
        + user_query
    )

    # Prefer environment variable; fall back to a local token.txt in project root (NOT committed)
    def _get_hf_token():
        t = os.environ.get("HF_TOKEN")
        if t:
            return t
        # look for token.txt in project root
        root = Path(__file__).resolve().parents[1]
        token_file = root / "token.txt"
        if token_file.exists():
            try:
                return token_file.read_text(encoding="utf-8").strip()
            except Exception:
                return None
        return None

    hf_token = _get_hf_token()

    def format_ok(text: str) -> bool:
        t = text.lower()
        return "concepts" in t and "practice questions" in t and "resources" in t

    def call_hf(prompt_text, max_tokens=256, temperature=0.0):
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
        payload = {"inputs": prompt_text, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    # Try HF API first if token is present
    if hf_token:
        try:
            data = call_hf(prompt, max_tokens=256, temperature=0.0)
            text = ""
            if isinstance(data, list) and data and "generated_text" in data[0]:
                text = data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                text = data["generated_text"]
            if not format_ok(text):
                data = call_hf(prompt, max_tokens=512, temperature=0.2)
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    text = data[0]["generated_text"]
                elif isinstance(data, dict) and "generated_text" in data:
                    text = data["generated_text"]
            if text and format_ok(text):
                return text
        except Exception:
            pass

    # Fallback to local generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    def local_generate(max_new_tokens=512, num_beams=4):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    out = local_generate(max_new_tokens=512, num_beams=4)
    if not format_ok(out):
        out = local_generate(max_new_tokens=768, num_beams=6)
    return out
