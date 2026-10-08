# Run locally

This guide runs the Next.js frontend, FastAPI backend, and PostgreSQL/pgvector database on your computer. Commands use Bash on Linux, macOS, or Windows through WSL2. On Windows, keep the checkout in the WSL filesystem and enable Docker Desktop's WSL integration.

## 1. Install prerequisites

- Git.
- Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- [Node.js](https://nodejs.org/) 24, including npm.
- [Docker Desktop](https://docs.docker.com/get-started/get-docker/) or Docker Engine with the Compose plugin. Start Docker before continuing.
- An [OpenRouter](https://openrouter.ai/) API key and access to your selected generation model.

A GPU is optional; the current embedding implementation uses the CPU. Allow several GB for Python dependencies and model assets, and extra disk space for your PDFs and PostgreSQL data. Initial installation and model downloads require Internet access.

Check your tools:

```bash
git --version
python3 --version
uv --version
node --version
npm --version
docker version
docker compose version
```

## 2. Clone and install

```bash
git clone https://github.com/Hari-jmr/gemma2.git
cd gemma2

# Override the cloud-specific cache paths checked into this repository.
# Repeat these exports in new terminals before running uv or npm.
export UV_CACHE_DIR="$HOME/.cache/uv"
export npm_config_cache="$HOME/.npm"

uv sync --locked --python 3.12
cd frontend
npm ci
cd ..
```

The repository's `uv.toml` and `frontend/.npmrc` use `/workspace` cache paths for the cloud machine. The exports above override them locally; you do not need to edit those files. `uv` creates the Python virtual environment automatically.

## 3. Start PostgreSQL with pgvector

From the repository root:

```bash
docker compose up -d --wait postgres
uv run python -m app.db
docker compose ps
```

The database listens on `127.0.0.1:5433`. The backend automatically uses:

```text
postgresql://rag@127.0.0.1:5433/rag
```

Schema initialization creates the `vector` extension, `rag_documents`, and `rag_chunks`. Embeddings use `vector(768)`, with cosine-distance retrieval. Initialization is safe to repeat.

Database files persist in `data/postgres/`; original PDFs are retained in other folders under `data/`. Do not delete this directory when updating the project. The supplied database uses passwordless authentication and is bound to loopback for trusted local development. An external or shared deployment needs database authentication and appropriate TLS configuration.

## 4. Download EmbeddingGemma 2 from Kaggle

Open the official [EmbeddingGemma 2 model page](https://www.kaggle.com/models/google/embeddinggemma-2). Review any access or license requirements shown there.

List the published variants:

```bash
uv run python scripts/kaggle_model.py
```

Choose a sentence-transformers-compatible variant and its actual version number. Download it using the full handle from Kaggle:

```bash
# Replace FRAMEWORK, VARIANT, and VERSION with the real published values.
uv run python scripts/kaggle_model.py \
  --handle google/embeddinggemma-2/FRAMEWORK/VARIANT/VERSION
```

The helper prints `EMBEDDING_MODEL_PATH=/absolute/path/to/model`. Copy that path into the `.env` file in step 6. It checks that the downloaded variant contains a sentence-transformers model directory. Do not use a GGUF or TFLite variant with this backend.

If Kaggle requires authentication, follow its current API authentication instructions or run the library's login flow in your local terminal:

```bash
uv run python -c "import kagglehub; kagglehub.login()"
```

Never commit Kaggle credentials. Keep the downloaded checkpoint at its versioned path; changing the embedding model requires re-indexing existing PDFs.

**Verification status:** the official model card supports sentence-transformers, but Kaggle variant discovery and the model download were blocked by the cloud environment's network policy. This guide therefore requires selecting an actual published handle rather than supplying an unverified one. Local model inference remains to be validated after downloading.

## 5. Prepare PDF extraction

The UI supports two parsers:

| Parser | Use it for | Extra model downloads |
| --- | --- | --- |
| PyMuPDF | PDFs with selectable text; fast initial testing | None |
| Docling | Scans, tables, and complex layouts; OCR | Layout, table, and RapidOCR assets |

To enable Docling, run from the repository root:

```bash
HF_HOME="$PWD/models/hf" HF_HUB_DISABLE_XET=1 \
  uv run docling-tools models download layout tableformer rapidocr \
  --output-dir "$PWD/models/docling"
```

Choose Docling and enable **Force OCR** for scanned pages or unreliable text layers. PyMuPDF mode extracts existing text only; it does not perform OCR in this application. Docling asset downloads and scanned-PDF conversion were blocked in cloud validation, so confirm extraction against your own scans locally.

## 6. Configure the backend

Create a file named `.env` in the repository root using your editor:

```dotenv
EMBEDDING_MODEL_PATH=/absolute/path/printed/by/the/kaggle/helper
OPENROUTER_API_KEY=replace-with-your-openrouter-key
LLM_MODEL=replace-with-an-available-openrouter-model-id
LLM_BASE_URL=https://openrouter.ai/api/v1
DATABASE_URL=postgresql://rag@127.0.0.1:5433/rag
```

Replace the placeholders. Use the exact model ID from OpenRouter's current catalog and ensure your account can use it. `.env` is ignored by Git. Keep the API key on the backend; do not add it to the frontend or a `NEXT_PUBLIC_*` variable.

The backend does **not** load `.env` automatically. The `uv run --env-file .env` commands below explicitly load it. If you configure a different database, initialize that database with `uv run --env-file .env python -m app.db` using an account with the required extension and schema privileges.

## 7. Run the backend

In terminal 1, from the repository root:

```bash
export UV_CACHE_DIR="$HOME/.cache/uv"
uv run --env-file .env uvicorn app.main:app \
  --host 127.0.0.1 --port 8000 --reload
```

Check it in another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expect `database_ready`, `embedding_model_configured`, and `llm_configured` to be `true` after configuration. These fields check database readiness and configuration presence; only uploading and asking a question proves the model and API actually work. FastAPI's interactive API documentation is at `http://127.0.0.1:8000/docs`.

## 8. Run the frontend

In terminal 2, from the repository root:

```bash
cd frontend
export npm_config_cache="$HOME/.npm"
npm run dev -- --hostname 127.0.0.1
```

Open **http://localhost:3000** in your browser. Next.js forwards `/api/*` requests to FastAPI on port 8000; no browser CORS setup is needed.

1. Select a PDF and choose Docling or PyMuPDF.
2. Click **Upload and index** and wait for the passage count.
3. Ask a question answered by the document.
4. Expand the source passages and check their page references.
5. Ask an unsupported question and check that the answer acknowledges insufficient evidence.

Only one uploaded document is searched per question. Questions and retrieved passages are sent to OpenRouter. This is a local prototype without application authentication; the commands above bind both servers to loopback.

## 9. Run checks or a production frontend build

With PostgreSQL running, from the repository root:

```bash
uv run pytest -q
cd frontend
npm run build
npm run typecheck
```

Nine tests passed in the cloud environment, including real PyMuPDF extraction and actual PostgreSQL/pgvector persistence and retrieval. The tests use a deterministic encoder and mocked LLM; they do not prove EmbeddingGemma or OpenRouter quality. Integration tests create and remove isolated schemas in the local database. Set `TEST_DATABASE_URL` if you need a separate test server.

To serve the built frontend, stop the development frontend first, then run:

```bash
# From frontend/; keep the backend and PostgreSQL running.
npm start -- --hostname 127.0.0.1
```

For a different backend address, set `BACKEND_URL` when starting development, or **before** the production build:

```bash
BACKEND_URL=http://127.0.0.1:8001 npm run build
npm start -- --hostname 127.0.0.1
```

## Stop and restart

Press **Ctrl+C** in each application terminal. From the repository root, stop PostgreSQL with:

```bash
docker compose stop postgres
```

To restart, run `docker compose up -d --wait postgres`, then restart the backend and frontend using steps 7 and 8. Stored documents and embeddings survive a database restart. The current UI does not list previous uploads, so keep a document ID for later API use or upload the PDF again.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Cache error mentioning `/workspace` | Export `UV_CACHE_DIR` and `npm_config_cache` as shown in step 2. |
| Docker cannot connect | Start Docker Desktop/Engine; on Windows, check WSL integration. |
| PostgreSQL unavailable or uninitialized | Check `docker compose ps`, then run `uv run python -m app.db`. Use `--env-file .env` for a custom database. |
| Port 5433 already in use | Change the host port in `compose.yaml` and match it in `DATABASE_URL`; set `TEST_DATABASE_URL` for tests. |
| Missing `EMBEDDING_MODEL_PATH` | Set the downloaded model directory in `.env` and start the backend with `--env-file .env`. |
| Wrong model format or dimension | Use the official sentence-transformers-compatible EmbeddingGemma 2 variant with 768-dimensional output. |
| No text extracted | Use Docling with Force OCR for scans; verify its asset download completed. |
| Model changed; re-index required | Upload the PDF again with the current model. |
| LLM API request failed | Verify the key, provider model ID, account access/credits, and Internet connectivity. |
| Upload times out | Try a smaller PDF or PyMuPDF for native text. The proxy allows five minutes; larger OCR jobs need a future background-worker implementation. |
| Frontend cannot reach backend | Check FastAPI is running and verify `/health`; match `BACKEND_URL` to its port. |

For database diagnostics, use `docker compose logs postgres`. Avoid posting logs that contain private document text or credentials.
