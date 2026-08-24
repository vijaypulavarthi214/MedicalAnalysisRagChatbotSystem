# Medical Document RAG

Upload a patient's medical PDF and ask questions about it. The backend extracts and
chunks the document, indexes it, retrieves relevant passages for each question, and
generates an answer grounded strictly in that document — never outside knowledge.

**Not HIPAA-compliant. Not a substitute for professional medical advice.** This is a
prototype application; do not use it with real patient data in a production or
clinical setting without a proper compliance review.

## Architecture

```
PDF upload
  → PyMuPDF text/layout extraction        (backend/app/ingestion/pdf_loader.py)
  → structure-aware section detection     (backend/app/ingestion/structure_parser.py)
  → chunking (~500 words, 10% overlap)    (backend/app/ingestion/chunker.py)
  → sentence-transformers embeddings      (backend/app/embeddings/embedding_service.py)
  → stored in ChromaDB                    (backend/app/retrieval/chroma_store.py)

Question
  → same embedding model embeds the query
  → hybrid search: semantic (Chroma) + BM25 (rank_bm25), top 10  (backend/app/retrieval/hybrid_search.py)
  → Cohere rerank narrows to top 3                                (backend/app/retrieval/reranker.py)
  → Groq LLM answers ONLY from those 3 chunks                     (backend/app/llm/groq_client.py)
  → answer + page/section citations returned to the frontend
```

Every stage is hand-written (no LangChain) so it's easy to read and test in isolation.
Set `DEBUG_RAG=true` to have `/query` include the raw hybrid-search and rerank results
plus per-stage latency in its response.

## Local development

```bash
cd backend
python3.12 -m venv ../.venv          # 3.12 recommended — chromadb/sentence-transformers
                                       # wheel support on 3.13+/3.14 can be unreliable
../.venv/bin/pip install -r requirements.txt
cp .env.example .env                  # fill in GROQ_API_KEY, COHERE_API_KEY
../.venv/bin/uvicorn app.main:app --reload
```

Run tests:

```bash
cd backend
../.venv/bin/python -m pytest -v
```

Serve the frontend against your local backend:

```bash
cd frontend
python3 -m http.server 5500
# edit config.js: API_BASE_URL = "http://localhost:8000"
# open http://localhost:5500
```

## Configuration

All config is via environment variables (see `backend/.env.example`). Never hardcode
API keys.

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key (free tier at console.groq.com/keys) |
| `GROQ_MODEL` | Groq model id, default `llama-3.3-70b-versatile` |
| `COHERE_API_KEY` | Cohere API key (for reranking) |
| `COHERE_RERANK_MODEL` | Default `rerank-v3.5` |
| `EMBEDDING_MODEL` | Hugging Face sentence-transformers model, default `all-MiniLM-L6-v2`. Used identically for indexing and querying. |
| `CHROMA_PERSIST_DIRECTORY` | Where Chroma stores its data |
| `HYBRID_ALPHA` | Semantic vs lexical weight in hybrid search (0-1, default 0.7) |
| `MAX_FILE_SIZE_MB` | Upload size cap, default 20 |
| `CORS_ORIGINS` | Comma-separated allowed origins. **Never `*` in production** — set to your exact Netlify domain. |
| `DEBUG_RAG` | `true` to include retrieval/rerank internals in `/query` responses |

## API

- `POST /upload` — multipart PDF upload. Returns `{document_id, filename, chunk_count, page_count, status}`.
- `GET /documents` — list uploaded documents.
- `DELETE /documents/{document_id}` — remove a document and its chunks.
- `POST /query` — `{question, document_id}` → `{answer, sources, document_id, debug}`. 404 if the document isn't found. If the answer isn't supported by the document, the fixed response `"This information was not found in the provided document."` is returned instead of a guess.
- `GET /health` — liveness/readiness probe.

## Deployment

### Backend → Render

1. Push this repo to GitHub.
2. In Render, create a new Blueprint from `render.yaml` (or a Web Service pointing at the root `Dockerfile`).
3. Set the secret env vars in the Render dashboard: `GROQ_API_KEY`, `COHERE_API_KEY`, `CORS_ORIGINS` (set this to your Netlify URL once you have it, e.g. `https://your-app.netlify.app`).
4. Render's free tier spins down after 15 minutes idle — the first request after that will be slow (cold start + model already baked into the image helps, but the instance itself still has to boot, often 30-60s+).
5. Free tier has no persistent disk, so `CHROMA_PERSIST_DIRECTORY` lives on ephemeral container storage — every restart/redeploy/spin-down wipes all uploaded documents. Re-upload after any of those. Upgrading to a paid plan + adding a `disk:` block to `render.yaml` fixes this if it becomes a problem.

### Frontend → Netlify

1. Edit `frontend/config.js` and set `API_BASE_URL` to your Render backend's URL.
2. In Netlify, create a new site from this repo with base directory `frontend` (or drag-and-drop the `frontend/` folder). `netlify.toml` inside `frontend/` handles the rest — no build step.
3. Once deployed, update `CORS_ORIGINS` on Render to the resulting Netlify URL and redeploy the backend.

## Testing

49 backend tests cover ingestion, embeddings, retrieval, reranking, the LLM client,
pipeline orchestration, and the API layer, with external services (Cohere, Groq,
Chroma) mocked or run against ephemeral in-memory instances. Run with:

```bash
cd backend && ../.venv/bin/python -m pytest -v
```

## Known limitations

- The document registry (`/documents` metadata) is in-memory and resets on backend
  restart — chunk data itself persists in Chroma, but the friendly filename/page-count
  listing does not survive a restart until re-uploaded.
- On Render's free tier there's no persistent disk, so Chroma's data is wiped on every
  restart/redeploy/idle spin-down too — treat the deployed app as demo-only unless you
  upgrade to a paid Render plan with an attached disk.
- `/upload` processes the PDF synchronously; very large documents will hold the
  request open for the full ingestion pipeline.
- Groq's free tier has per-minute/per-day rate limits that can be hit under heavy
  testing — if `/query` starts failing with a rate-limit error, wait a bit or check
  your usage at console.groq.com.
