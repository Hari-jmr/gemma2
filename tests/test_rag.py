import io
import os
import uuid

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from reportlab.pdfgen import canvas

from app import db, main
from app.extraction import chunk_text, extract_pdf


class Encoder:
    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), db.DIMENSIONS), dtype='float32')
        for n, text in enumerate(texts):
            vectors[n, 0 if 'Paris' in text or 'capital' in text else 1] = 1
        return vectors


def sample_pdf():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, 'Paris is the capital of France.')
    pdf.showPage()
    pdf.drawString(72, 720, 'Bananas are yellow.')
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
def database(monkeypatch):
    # Use an isolated schema, never remove user document tables.
    url = os.getenv('TEST_DATABASE_URL', db.DEFAULT_DATABASE_URL)
    schema = 'rag_test_' + uuid.uuid4().hex
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    from psycopg.conninfo import make_conninfo
    monkeypatch.setenv('DATABASE_URL', make_conninfo(url, options=f'-c search_path={schema},public'))
    try:
        db.initialize()
        yield
    finally:
        with psycopg.connect(url, autocommit=True) as connection:
            connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


def test_real_pdf_postgres_retrieval_and_chat(tmp_path, monkeypatch, database):
    monkeypatch.setattr(main, 'DATA', tmp_path)
    monkeypatch.setattr(main, 'encoder', Encoder())
    client = TestClient(main.app)
    upload = client.post('/documents?parser=pymupdf', files={'file': ('sample.pdf', sample_pdf(), 'application/pdf')})
    assert upload.status_code == 200, upload.text
    assert upload.json()['chunks'] == 2
    query = {'document_id': upload.json()['document_id'], 'question': 'What is the capital?', 'top_k': 1}
    source = client.post('/retrieve', json=query).json()['sources'][0]
    assert source['page'] == 1 and 'Paris' in source['text']
    assert source['score'] == pytest.approx(1)
    monkeypatch.setenv('LLM_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_MODEL', 'test-model')
    monkeypatch.setenv('LLM_BASE_URL', 'https://openrouter.ai/api/v1')
    def api(url, **kwargs):
        assert url == 'https://openrouter.ai/api/v1/chat/completions'
        assert 'Paris' in kwargs['json']['messages'][1]['content']
        import httpx
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Paris [1].'}}]}, request=httpx.Request('POST', url))
    monkeypatch.setattr(main.httpx, 'post', api)
    response = client.post('/chat', json=query)
    assert response.status_code == 200
    assert response.json()['answer'] == 'Paris [1].'
    assert response.json()['sources'][0]['page'] == 1
    # Reconnect and retrieve persisted rows from PostgreSQL.
    assert client.post('/retrieve', json=query).json()['sources'][0] == source
    assert client.get('/health').json()['database_ready'] is True


def test_document_isolation_and_model_change(database):
    vectors = Encoder().encode(['Paris', 'Bananas'])
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    db.save_document(first, 'first.pdf', 'pymupdf', 'model-a', [{'page': 1, 'text': 'Paris'}], vectors[:1])
    db.save_document(second, 'second.pdf', 'pymupdf', 'model-a', [{'page': 2, 'text': 'Bananas'}], vectors[1:])
    results = db.search(second, vectors[0], 10, 'model-a')
    assert len(results) == 1 and results[0]['filename'] == 'second.pdf'
    with pytest.raises(ValueError, match='model changed'):
        db.search(first, vectors[0], 1, 'model-b')
    with pytest.raises(LookupError):
        db.search(uuid.uuid4().hex, vectors[0], 1, 'model-a')


def test_save_is_transactional(database):
    document_id = uuid.uuid4().hex
    vector = Encoder().encode(['Paris'])
    with pytest.raises(psycopg.errors.NotNullViolation):
        db.save_document(document_id, 'bad.pdf', 'pymupdf', 'model-a', [{'page': 1, 'text': None}], vector)
    with pytest.raises(LookupError):
        db.search(document_id, vector[0], 1, 'model-a')


def test_invalid_vectors():
    for vector in [np.ones(2), np.zeros(768), np.full(768, np.nan)]:
        with pytest.raises(ValueError):
            db.checked_vector(vector)


def test_invalid_upload_and_document_id(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'ready', lambda: True)
    monkeypatch.setattr(main, 'DATA', tmp_path)
    monkeypatch.setattr(main, 'encoder', Encoder())
    client = TestClient(main.app)
    assert client.post('/documents', files={'file': ('x.pdf', b'bad')}).status_code == 400
    assert not list(tmp_path.iterdir())
    assert client.post('/retrieve', json={'document_id': '../secret', 'question': 'hello'}).status_code == 422
    assert client.post('/documents?parser=pymupdf&force_ocr=true', files={'file': ('x.pdf', sample_pdf())}).status_code == 400
    assert client.post('/documents?parser=invalid', files={'file': ('x.pdf', sample_pdf())}).status_code == 422


def test_missing_model_is_explicit(monkeypatch):
    monkeypatch.setattr(db, 'ready', lambda: True)
    monkeypatch.setattr(main, 'encoder', None)
    monkeypatch.delenv('EMBEDDING_MODEL_PATH', raising=False)
    client = TestClient(main.app)
    assert client.post('/documents', files={'file': ('x.pdf', sample_pdf())}).status_code == 503


def test_database_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(db, 'ready', lambda: False)
    response = TestClient(main.app).post('/documents', files={'file': ('x.pdf', sample_pdf())})
    assert response.status_code == 503 and 'PostgreSQL' in response.json()['detail']


def test_pymupdf_page_numbers_and_ocr_constraint(tmp_path):
    path = tmp_path / 'sample.pdf'
    path.write_bytes(sample_pdf())
    chunks = extract_pdf(path, parser='pymupdf')
    assert chunks[0]['page'] == 1 and 'Paris' in chunks[0]['text']
    assert chunks[1]['page'] == 2 and 'Bananas' in chunks[1]['text']
    with pytest.raises(ValueError, match='OCR'):
        extract_pdf(path, force_ocr=True, parser='pymupdf')


def test_chunk_overlap():
    assert [len(c) for c in chunk_text('x' * 3000)] == [1600, 1600, 200]
    assert chunk_text('   ') == []
