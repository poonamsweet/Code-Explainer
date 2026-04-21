# AI Codebase Explainer (Streamlit)

Upload a code file, preview it, generate an AI explanation, and ask questions about the code using embeddings-based retrieval (SentenceTransformers + ChromaDB) as context for OpenAI.

## Features

- **File upload + preview**: upload a code file and view it in the UI.
- **Explain**: get a clear explanation of the uploaded code via OpenAI.
- **Ask questions**: ask a question; the app retrieves the most relevant code chunks from ChromaDB and sends **context + question** to OpenAI.
- **Embeddings + storage**:
  - Chunking utilities (including 500–800 char chunks)
  - SentenceTransformers embeddings
  - Persistent ChromaDB storage in `.chroma/`

## Project structure

- `app.py`: Streamlit UI (upload, explain, Q&A).
- `ai_helper.py`: OpenAI client helpers + prompt logic + error handling.
- `embeddings.py`: chunking, embedding generation, ChromaDB storage + similarity search.
- `requirements.txt`: Python dependencies.

## Setup

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Configure OpenAI API key

Create a file named `.env` next to `app.py`:

```env
OPENAI_API_KEY=YOUR_REAL_KEY_HERE
```

Notes:
- **Do not** wrap the key in quotes.
- The `.env` file is ignored by git (`.gitignore`) to avoid committing secrets.

### 3) Run the app

```bash
streamlit run app.py
```

## Using the app

1) Upload a code file
2) Use the **Explain** tab to generate an explanation
3) Use the **Chat** tab:
   - Enter a question in the text box
   - Click **Ask**
   - The app retrieves relevant chunks (top_k) and answers using only the provided context

## Configuration (optional)

Environment variables:
- `OPENAI_MODEL`: OpenAI model name (default: `gpt-4.1-mini`)
- `EMBED_MODEL`: SentenceTransformers model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `CHROMA_DIR`: Chroma persistence directory (default: `.chroma`)
- `CHROMA_COLLECTION`: Chroma collection name (default: `code_chunks`)

## Troubleshooting

- **Authentication failed**:
  - Ensure `OPENAI_API_KEY` is set in `.env` and you restarted Streamlit.
  - A placeholder key (example-looking value) will always fail.
