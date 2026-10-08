"""Transactional document storage and exact cosine retrieval with pgvector."""
import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

DEFAULT_DATABASE_URL = "postgresql://rag@127.0.0.1:5433/rag"
DIMENSIONS = 768


def connect():
    connection = psycopg.connect(os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL), connect_timeout=5)
    try:
        register_vector(connection)
    except Exception:
        connection.close()
        raise
    return connection


def initialize():
    with psycopg.connect(os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL), connect_timeout=5) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute("""CREATE TABLE IF NOT EXISTS rag_documents (
            id UUID PRIMARY KEY,
            filename TEXT NOT NULL,
            parser TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""")
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


def save_document(document_id, filename, parser, embedding_model, chunks, vectors):
    if len(chunks) != len(vectors):
        raise ValueError("Chunk count does not match embedding count.")
    rows = [(document_id, n, chunk["page"], chunk["text"], checked_vector(vector))
            for n, (chunk, vector) in enumerate(zip(chunks, vectors))]
    with connect() as connection:
        connection.execute("INSERT INTO rag_documents (id, filename, parser, embedding_model) VALUES (%s, %s, %s, %s)",
                           (document_id, filename, parser, embedding_model))
        with connection.cursor() as cursor:
            cursor.executemany("INSERT INTO rag_chunks (document_id, ordinal, page, content, embedding) VALUES (%s, %s, %s, %s, %s)", rows)


def search(document_id, vector, top_k, embedding_model):
    vector = checked_vector(vector)
    with connect() as connection:
        document = connection.execute("SELECT filename, embedding_model FROM rag_documents WHERE id = %s", (document_id,)).fetchone()
        if document is None:
            raise LookupError("Document not found or indexing incomplete.")
        if document[1] != embedding_model:
            raise ValueError("The embedding model changed. Re-index this PDF before searching.")
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("""SELECT page, content AS text, 1 - (embedding <=> %s) AS score
                FROM rag_chunks WHERE document_id = %s
                ORDER BY embedding <=> %s, ordinal LIMIT %s""", (vector, document_id, vector, top_k))
            return [{**row, "score": float(row["score"]), "filename": document[0], "source": n + 1}
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
