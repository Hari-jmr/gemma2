"""Transactional document storage and exact cosine retrieval with pgvector."""
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

DIMENSIONS = 768
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def database_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not configured. Set it in the .env file.")
    return url


def connect():
    connection = psycopg.connect(database_url(), connect_timeout=5)
    try:
        register_vector(connection)
    except Exception:
        connection.close()
        raise
    return connection


def initialize():
    with psycopg.connect(database_url(), connect_timeout=5) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute("""CREATE TABLE IF NOT EXISTS rag_documents (
            id UUID PRIMARY KEY,
            filename TEXT NOT NULL,
            parser TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""")
        connection.execute("ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS sha256 TEXT")
        connection.execute("CREATE INDEX IF NOT EXISTS rag_documents_sha256_idx ON rag_documents (sha256)")
        connection.execute(f"""CREATE TABLE IF NOT EXISTS rag_chunks (
            document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            page INTEGER,
            content TEXT NOT NULL,
            embedding vector({DIMENSIONS}) NOT NULL,
            PRIMARY KEY (document_id, ordinal)
        )""")


def checked_vector(vector):
    result = np.asarray(vector, dtype="float32")
    if result.shape != (DIMENSIONS,) or not np.isfinite(result).all() or np.linalg.norm(result) == 0:
        raise ValueError(f"Expected a finite, nonzero {DIMENSIONS}-dimensional embedding.")
    return result


def find_document_by_hash(sha256):
    with connect() as connection:
        row = connection.execute("SELECT id FROM rag_documents WHERE sha256 = %s", (sha256,)).fetchone()
        return str(row[0]) if row else None


def save_document(document_id, filename, parser, embedding_model, chunks, vectors, sha256=None):
    if len(chunks) != len(vectors):
        raise ValueError("Chunk count does not match embedding count.")
    rows = [(document_id, n, chunk["page"], chunk["text"], checked_vector(vector))
            for n, (chunk, vector) in enumerate(zip(chunks, vectors))]
    with connect() as connection:
        connection.execute("DELETE FROM rag_documents WHERE id = %s", (document_id,))
        connection.execute("INSERT INTO rag_documents (id, filename, parser, embedding_model, sha256) VALUES (%s, %s, %s, %s, %s)",
                           (document_id, filename, parser, embedding_model, sha256))
        with connection.cursor() as cursor:
            cursor.executemany("INSERT INTO rag_chunks (document_id, ordinal, page, content, embedding) VALUES (%s, %s, %s, %s, %s)", rows)


def search(document_ids, vector, top_k, embedding_model):
    if isinstance(document_ids, str):
        document_ids = [document_ids]
    requested = list(dict.fromkeys(str(item) for item in document_ids))
    vector = checked_vector(vector)
    with connect() as connection:
        documents = connection.execute("SELECT id, filename, embedding_model FROM rag_documents WHERE id = ANY(%s::uuid[])", (requested,)).fetchall()
        names = {str(row[0]): row[1] for row in documents}
        if len(names) != len(requested):
            raise LookupError("Document not found or indexing incomplete.")
        if any(row[2] != embedding_model for row in documents):
            raise ValueError("The embedding model changed. Re-index these PDFs before searching.")
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("""SELECT c.document_id, c.page, c.content AS text, 1 - (c.embedding <=> %s) AS score
                FROM rag_chunks c WHERE c.document_id = ANY(%s::uuid[])
                ORDER BY c.embedding <=> %s, c.ordinal LIMIT %s""", (vector, requested, vector, top_k))
            return [{**row, "document_id": str(row["document_id"]), "score": float(row["score"]),
                     "filename": names[str(row["document_id"])], "source": n + 1}
                    for n, row in enumerate(cursor.fetchall())]


def ready():
    try:
        with connect() as connection:
            return connection.execute("SELECT to_regclass('rag_documents'), to_regclass('rag_chunks')").fetchone() == ("rag_documents", "rag_chunks")
    except psycopg.Error:
        return False


if __name__ == "__main__":
    initialize()
    print("PostgreSQL tables and pgvector extension initialized.")
