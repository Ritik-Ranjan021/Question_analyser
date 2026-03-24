import os
import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from utils import load_text_from_file
from openai import OpenAI


st.set_page_config(page_title="Question RAG Explorer", layout="wide")

DATA_DIR = "data"
DEFAULT_INDEX = os.path.join(DATA_DIR, "question_index.faiss")
DEFAULT_META = os.path.join(DATA_DIR, "questions_meta.jsonl")


@st.cache_resource
def load_index(idx_path):
    if not os.path.exists(idx_path):
        return None
    index = faiss.read_index(idx_path)
    return index


@st.cache_data
def load_metadata(meta_path):
    if not os.path.exists(meta_path):
        return []
    entries = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


@st.cache_resource
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)


def search(index, model, entries, query, k=5):
    if index is None:
        return []
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(entries):
            continue
        results.append(entries[idx])
    return results


def call_groq(api_key, prompt, model="llama-3.3-70b"):
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer using the context only. If not found, say 'Not in document'."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return resp.choices[0].message.content


def main():
    st.title("Question RAG Explorer")

    st.sidebar.header("Settings")
    index_path = st.sidebar.text_input("Question index path", DEFAULT_INDEX)
    meta_path = st.sidebar.text_input("Metadata JSONL path", DEFAULT_META)
    groq_key = st.sidebar.text_input("Groq API Key", type="password")
    groq_model = st.sidebar.text_input("Groq model", "llama-3.3")
    model_name = st.sidebar.text_input("Embedding model", "all-MiniLM-L6-v2")

    index = load_index(index_path)
    entries = load_metadata(meta_path)
    embed_model = load_embed_model(model_name)

    q = st.text_input("Ask a question:")
    top_k = st.slider("Top k", 1, 10, 5)

    if q:
        with st.spinner("Searching..."):
            hits = search(index, embed_model, entries, q, k=top_k)

        st.subheader("Top matching questions")
        for h in hits:
            st.markdown(f"**Source:** {h.get('source')}  — **Year:** {h.get('year')}")
            st.write(h.get("text")[:1000])

        st.subheader("Generate suggestion")
        context = "\n\n".join([h.get("text") for h in hits])
        prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nProvide: 1) short answer, 2) key topics, 3) 3 practice questions, 4) resources (links if available)."

        if not groq_key:
            st.warning("Provide Groq API Key in the sidebar to generate answers.")
        else:
            if st.button("Generate via Groq"):
                with st.spinner("Calling Groq..."):
                    try:
                        answer = call_groq(groq_key, prompt, model=groq_model)
                        st.markdown("**LLM Answer**")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")


if __name__ == "__main__":
    main()
