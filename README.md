# PDF RAG with EmbeddingGemma 2

A PDF question-answering prototype with a **Next.js frontend**, **uv + FastAPI backend**, **PostgreSQL + pgvector storage**, and **OpenRouter** for answer generation.

See **[deploy.md](deploy.md)** for complete local installation, model downloads, configuration, startup, and troubleshooting. It covers Linux, macOS, and Windows through WSL2, including overrides for this repository's cloud-specific cache paths.

## Current features

- Upload a PDF and choose Docling or PyMuPDF for extraction.
- Use Docling with RapidOCR for scanned pages, tables, and complex layouts.
- Use PyMuPDF for fast extraction of existing PDF text.
- Preserve page references and split text into overlapping passages.
- Generate local, normalized 768-dimensional EmbeddingGemma 2 embeddings.
- Persist documents, chunks, and embeddings in PostgreSQL; retrieve passages using pgvector cosine distance within the selected document.
- Send retrieved passages and the question to an OpenRouter-compatible LLM.
- Display answers and expandable source passages in the Next.js UI.

The earlier static HTML frontend and FAISS storage have been replaced. Embedding and live LLM behavior still need validation with downloaded model assets and working provider access; see the validation status below.

## Architecture

```text
PDF upload in Next.js
        |
        v
FastAPI -> Docling/RapidOCR or PyMuPDF -> page-aware text chunks
        |
        v
EmbeddingGemma 2 -> PostgreSQL/pgvector

Question -> query embedding -> cosine retrieval -> OpenRouter
        |
        v
Answer and source passages in Next.js
```

## Local quick start

Install Git, Python 3.12, uv, Node.js 24, and Docker with Compose. Start Docker, then:

```bash
git clone https://github.com/Hari-jmr/gemma2.git
cd gemma2
export UV_CACHE_DIR="$HOME/.cache/uv"
export npm_config_cache="$HOME/.npm"
uv sync --locked --python 3.12
docker compose up -d --wait postgres
uv run python -m app.db
```

Before starting the backend, follow [deploy.md](deploy.md) to download a compatible official Kaggle EmbeddingGemma 2 variant, prepare Docling assets if using OCR, and create a root `.env` file:

| Variable | Purpose |
| --- | --- |
| `EMBEDDING_MODEL_PATH` | Absolute path to the downloaded sentence-transformers model directory |
| `OPENROUTER_API_KEY` | Backend-only OpenRouter credential |
| `LLM_MODEL` | Available OpenRouter model ID |
| `LLM_BASE_URL` | Optional; defaults to `https://openrouter.ai/api/v1` |
| `DATABASE_URL` | Optional; defaults to `postgresql://rag@127.0.0.1:5433/rag` |

Terminal 1, from the repository root:

```bash
export UV_CACHE_DIR="$HOME/.cache/uv"
uv run --env-file .env uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2, from the repository root:

```bash
cd frontend
export npm_config_cache="$HOME/.npm"
npm ci
npm run dev -- --hostname 127.0.0.1
```

Open **http://localhost:3000** locally. API documentation is at **http://127.0.0.1:8000/docs**. `.env` is not loaded automatically; the backend command explicitly loads it. Never put API keys in frontend variables or commit them.

Next.js forwards `/api/*` to FastAPI. Set `BACKEND_URL` to use a different backend address; production builds need this setting before `npm run build`. The upload limit is 50 MB and the proxy timeout is five minutes. Larger OCR jobs will require background processing.

## PDF parser choice

| Parser | Best use | OCR in this app |
| --- | --- | --- |
| Docling (default) | Scanned PDFs, tables, complex layouts | RapidOCR; optional Force OCR |
| PyMuPDF | Native PDFs with selectable text | None |

PyMuPDF with Force OCR is rejected. Choose Docling for scans. Verify extraction and reading order against your PDFs, especially figures, handwriting, and unusual layouts. PyMuPDF is AGPL/commercial licensed; review its terms before distributing the application.

## Database and persistence

Docker Compose runs digest-pinned PostgreSQL 17 with pgvector, published on `127.0.0.1:5433`. It uses passwordless authentication for trusted local development. Shared deployments need appropriate authentication and TLS.

- `rag_documents`: document ID, filename, parser, model identity, and creation time.
- `rag_chunks`: document ID, passage order, page, text, and `vector(768)` embedding.
- Writes are transactional; retrieval uses exact cosine-distance ranking filtered by document ID.
- A change to the versioned model path requires re-indexing; keep checkpoint contents fixed at that path.

Database files persist in ignored `data/postgres/`; original PDFs remain under `data/`. Stop PostgreSQL with `docker compose stop postgres` without deleting retained data. Existing FAISS outputs are not migrated automatically; upload those PDFs again. The current UI searches one document at a time and does not list previous uploads; retain document IDs for later API use.

## API

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Database readiness and model/LLM configuration presence |
| `POST /documents?parser=docling&force_ocr=false` | Multipart PDF upload, extraction, embedding, and database storage; `parser=pymupdf` selects native text extraction |
| `POST /retrieve` | Retrieve source passages without generating an answer |
| `POST /chat` | Generate an answer using retrieved passages |

Retrieval and chat accept JSON with `document_id`, `question`, and optional `top_k` (1–10, default 5). Questions and retrieved passages are sent to the configured provider. This prototype has no application authentication and is intended for trusted development.

## Validation status

**Verified:** nine tests pass, including real PyMuPDF extraction, page numbers, actual PostgreSQL/pgvector writes and retrieval, document isolation, transactional rollback, model mismatch, invalid embeddings, and API errors. PostgreSQL restart persistence, the Next.js production build, TypeScript checks, and frontend-to-backend requests also pass.

**Pending:** the tests use a deterministic encoder and mocked LLM. The actual EmbeddingGemma 2 checkpoint has not been downloaded, and Docling OCR, model retrieval quality, and live OpenRouter generation remain unverified. In cloud onboarding, `api.kaggle.com`, `openrouter.ai`, and `us.aws.cdn.hf.co` returned network-policy denials; the required configuration changes were saved for review. These cloud restrictions do not establish an access problem on your local computer. Kaggle variant/access requirements and the chosen OpenRouter model must be confirmed locally.

With PostgreSQL running:

```bash
uv run pytest -q
cd frontend
npm run build
npm run typecheck
```

Integration tests use temporary isolated schemas; `TEST_DATABASE_URL` overrides the local test connection. For a production frontend build, stop the development frontend and run `npm start -- --hostname 127.0.0.1` after `npm run build`, keeping FastAPI and PostgreSQL running.

## Code structure

```text
frontend/
  app/page.tsx           PDF upload, parser choice, questions and sources
  app/layout.tsx         Layout and metadata
  app/globals.css        Responsive styles
  next.config.ts         FastAPI proxy
  package.json           Frontend scripts and dependencies
  package-lock.json      Locked npm dependencies
app/
  main.py                FastAPI routes, embedding inference and LLM calls
  db.py                  PostgreSQL schema, persistence and retrieval
  extraction.py          Docling/PyMuPDF extraction and chunking
compose.yaml             Local PostgreSQL + pgvector service
scripts/kaggle_model.py  Official Kaggle variant discovery/download helper
tests/test_rag.py        PDF, PostgreSQL and API tests
pyproject.toml           Python dependencies
uv.lock                  Locked Python dependencies
deploy.md                Complete local setup and troubleshooting guide
```

Generated dependencies, build outputs, model downloads, PDFs, database files, and `.env` are excluded from Git.
