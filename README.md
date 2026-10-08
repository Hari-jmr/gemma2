# PDF RAG prototype

Next.js (TypeScript/React) frontend and a uv/FastAPI backend. PDFs are parsed by Docling/RapidOCR or PyMuPDF, embedded locally with EmbeddingGemma 2, stored in PostgreSQL/pgvector, and answered through OpenRouter. Node 24 and Python 3.12 are validated.

## Start the database and backend

From `/workspace/gemma2`:

```sh
uv sync --locked
docker compose up -d --wait postgres
uv run python -m app.db
export EMBEDDING_MODEL_PATH=/absolute/path/to/downloaded/model
export LLM_MODEL=your-provider-model-id
# Supply OPENROUTER_API_KEY securely through environment settings.
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker Compose uses a digest-pinned PostgreSQL 17 image with pgvector. It persists database files in ignored `data/postgres/`, publishes only on `127.0.0.1:5433`, and uses passwordless authentication for trusted local development. The default database URL is `postgresql://rag@127.0.0.1:5433/rag`. For a managed database, securely configure `DATABASE_URL` and initialize the schema with an account permitted to create the vector extension and tables. Use authenticated, TLS-secured connections for deployment. Never put credentials in source files.

`uv run python -m app.db` is repeatable: it enables the extension and creates `rag_documents` and `rag_chunks`. Chunks store text, page numbers, and `vector(768)` embeddings. Document rows track filename, parser, and the versioned model path. Writes are transactional. Retrieval uses pgvector cosine distance and exact nearest-neighbor search filtered by document ID; this is appropriate for a small PDF prototype. A model-path change requires re-indexing. Keep the checkpoint contents fixed at their versioned path. Consider an approximate vector index after measuring a larger workload.

FAISS is no longer used. Any earlier local FAISS files are left untouched; PDFs must be re-indexed into PostgreSQL. `docker compose stop postgres` stops the database without deleting its retained data. Do not delete `data/postgres/` when refreshing dependencies.

## Start the frontend

In a second terminal:

```sh
cd /workspace/gemma2/frontend
npm ci
npm run dev
```

Next.js runs on port 3000 and proxies `/api/*` to FastAPI on port 8000. `BACKEND_URL` can override the backend address; for production builds, set it before `npm run build`. API keys stay in the FastAPI environment. For production: `npm run build` then `npm start`. Run `npm run typecheck` after dev/build generates route types. The upload proxy supports the backend's 50 MB limit, with a five-minute timeout for CPU OCR. Longer jobs will need a background worker.

## PDF extraction

Choose a parser in the upload UI or use `POST /documents?parser=docling|pymupdf&force_ocr=false`.

| Parser | Best use | OCR |
| --- | --- | --- |
| Docling (default) | Scanned PDFs, tables, complex layouts | RapidOCR enabled; optional force OCR |
| PyMuPDF | Fast extraction from PDFs with an existing text layer | Text extraction only in this app |

Both preserve page numbers and split page text into overlapping passages. PyMuPDF with force OCR is rejected; choose Docling for scans. Figures, handwriting, unusual layouts, and reading order still need evaluation against your PDFs. PyMuPDF is AGPL/commercial licensed; review its terms before distributing the application.

Download Docling assets before using its parser:

```sh
HF_HOME=/workspace/gemma2/models/hf HF_HUB_DISABLE_XET=1 uv run docling-tools models download layout tableformer rapidocr --output-dir /workspace/gemma2/models/docling
```

## Embedding model and LLM

The official EmbeddingGemma 2 model card supports sentence-transformers. The app loads its text-only encoder in float32 with Document and SearchQuery prompts, normalized 768-dimensional output, local files only, and remote code disabled. It does not substitute another embedding model.

Run `uv run python scripts/kaggle_model.py` to list official Kaggle variants. Select a versioned transformers-compatible variant, then run:

```sh
uv run python scripts/kaggle_model.py --handle google/embeddinggemma-2/FRAMEWORK/VARIANT/VERSION
```

Use the printed local directory for `EMBEDDING_MODEL_PATH`. This handle must come from Kaggle's returned variants, not a guessed identifier. Configure `OPENROUTER_API_KEY` securely and `LLM_MODEL` with an available OpenRouter model ID. `LLM_BASE_URL` defaults to `https://openrouter.ai/api/v1`.

Remaining access blockers from onboarding: `api.kaggle.com`, `openrouter.ai`, and the Docling artifact host `us.aws.cdn.hf.co` returned network-policy denials. Their network additions and the OpenRouter key requirement are saved in the environment draft. Review and save these in environment settings before retrying. No embedding checkpoint has been downloaded. Docling OCR, actual EmbeddingGemma retrieval quality, and live LLM generation are unverified. Kaggle authentication requirements and the suggested OpenRouter model `openai/gpt-4.1-mini` remain to be confirmed after access works.

## Validation

With PostgreSQL running:

```sh
uv run pytest -q
cd frontend
npm run build
npm run typecheck
```

Nine tests cover real PyMuPDF extraction, page numbers, actual PostgreSQL/pgvector writes and retrieval, document isolation, transactional rollback, model mismatch, invalid embeddings, and API errors. An isolated database schema is created and removed per integration test; `TEST_DATABASE_URL` overrides the local test database connection. The encoder is deterministic and the LLM API is mocked in tests. The PostgreSQL restart persistence check was also verified against a temporary record and cleaned up afterward. Next.js production build and TypeScript checks pass.

API: `GET /health` (including database readiness), `POST /documents` (multipart PDF), `POST /retrieve`, and `POST /chat` (JSON document_id, question, optional top_k). Only one document is searched per question. Source passages and embeddings persist in PostgreSQL; original PDFs remain in ignored `data/` folders. Retain the document ID to use the API after a restart. The provider receives questions and relevant passages. This prototype has no application authentication and is intended for trusted development.

## Code structure

```text
frontend/
  app/page.tsx          PDF upload, parser choice, chat, source passages
  app/layout.tsx       Layout and metadata
  app/globals.css      Responsive styles
  next.config.ts       FastAPI proxy
  package.json         Frontend scripts and dependencies
  package-lock.json    Locked npm dependencies
app/
  main.py              FastAPI routes, embeddings and LLM calls
  db.py                PostgreSQL schema, persistence and cosine retrieval
  extraction.py        Docling/PyMuPDF extraction and chunking
compose.yaml           Local PostgreSQL + pgvector service
scripts/kaggle_model.py Official model discovery and download helper
tests/test_rag.py       PDF, PostgreSQL and API tests
pyproject.toml          Python dependencies
uv.lock                Locked Python dependencies
```

Generated `.venv/`, `frontend/node_modules/`, `frontend/.next/`, `models/`, and `data/` are ignored by Git. Only source code and dependency lockfiles belong in Git; local documents, database files, model downloads, and credentials stay out of the repository.
