import argparse
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rag import load_index, retrieve, generate_suggestions
import argparse
import os


def load_index(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def retrieve(index, meta, model, query, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(meta):
            continue
        results.append(meta[idx]["text"])
    return results


def generate_suggestions(retrieved_texts, user_query):
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

    hf_token = os.environ.get("HF_TOKEN")
    model_name = "google/flan-t5-base"

    def format_ok(text: str) -> bool:
        t = text.lower()
        return "concepts" in t and "practice questions" in t and "resources" in t
    def call_hf(max_tokens=256, temperature=0.0):
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    # Try up to 2 attempts: first attempt conservative, second attempt more generous
    if hf_token:
        try:
            data = call_hf(max_tokens=256, temperature=0.0)
            text = ""
            if isinstance(data, list) and data and "generated_text" in data[0]:
                text = data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                text = data["generated_text"]
            if not format_ok(text):
                # retry with larger budget and slight randomness
                data = call_hf(max_tokens=512, temperature=0.2)
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    text = data[0]["generated_text"]
                elif isinstance(data, dict) and "generated_text" in data:
                    text = data["generated_text"]
            if text and format_ok(text):
                return text
        except Exception:
            pass

    # Fallback: local generation using installed transformers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    # Local generation with up to 2 attempts
    def local_generate(max_new_tokens=512, num_beams=4):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    out = local_generate(max_new_tokens=512, num_beams=4)
    if not format_ok(out):
        out = local_generate(max_new_tokens=768, num_beams=6)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="User query (put in quotes)")
    parser.add_argument("--index_path", default="data/index.faiss")
    parser.add_argument("--meta_path", default="data/metadata.pkl")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    user_query = " ".join(args.query)
    index, meta = load_index(args.index_path, args.meta_path)
    if index is None:
        print("Index not found. Run build_index first.")
        return
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    retrieved = retrieve(index, meta, embed_model, user_query, top_k=args.top_k)
    if not retrieved:
        print("No relevant passages found.")
        return

    print("\n--- Retrieved passages (for debugging) ---\n")
    for i, p in enumerate(retrieved, 1):
        print(f"[{i}]", p[:400].replace('\n', ' '))
        print("----------------------------------------------------")

    out = generate_suggestions(retrieved, user_query)
    print("\n=== Suggestions ===\n")
    print(out)


if __name__ == "__main__":
    main()
