Local RAG pipeline for past-year questions

Quick start

1. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "requirements.txt"
```

2. Put your question files (PDF or .txt) into the `data/` folder.

3. Build the index:

```powershell
python src/build_index.py --data_dir data --index_path data/index.faiss --meta_path data/metadata.pkl
```

4. Query the index:

```powershell
python src/query.py "suggest important concepts for these past-year questions"
```

Optional: use the Hugging Face Inference API for generation (recommended to avoid large local downloads).

1. Create a Hugging Face token at https://huggingface.co/settings/tokens and set it in PowerShell:

```powershell
$env:HF_TOKEN = "hf_xxx"
```

2. Run the same `python src/query.py ...` command; the script will call the HF Inference API when `HF_TOKEN` is present.

Alternative (local file): create a file named `token.txt` in the project root containing your Hugging Face token on a single line. The app will read this file at runtime if `HF_TOKEN` is not set in the environment. Do NOT commit `token.txt` — it is listed in `.gitignore`.

Run a simple web UI (after building the index):

```powershell
pip install -r "requirements.txt"
# start the backend
python src/server.py
```

Open http://localhost:8000/static in your browser and use the form to query.

Files
- `src/build_index.py`: ingest files and build FAISS index
- `src/query.py`: run a retrieval and generate suggestions using a local T5 model
- `src/utils.py`: helpers for loading and chunking text
