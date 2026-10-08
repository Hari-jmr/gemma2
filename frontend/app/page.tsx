"use client";

import { useState } from "react";

type Source = { source: number; filename: string; page: number | null; text: string; score: number };

async function request<T>(path: string, options: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(typeof data?.detail === "string" ? data.detail : `Request failed (${response.status}). Please try again.`);
  if (!data) throw new Error("The server returned an unexpected response.");
  return data as T;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [forceOcr, setForceOcr] = useState(false);
  const [parser, setParser] = useState<"docling" | "pymupdf">("docling");
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("Upload a PDF to get started.");
  const [busy, setBusy] = useState<"upload" | "chat" | null>(null);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);

  async function upload(event: React.FormEvent) {
    event.preventDefault();
    if (!file || busy) return;
    if (file.size > 50 * 1024 * 1024) { setStatus("Please choose a PDF smaller than 50 MB."); return; }
    setBusy("upload"); setDocumentId(null); setAnswer(""); setSources([]);
    setStatus("Extracting text and creating embeddings. This may take a few minutes…");
    try {
      const body = new FormData(); body.append("file", file);
      const data = await request<{ document_id: string; chunks: number }>(`/documents?force_ocr=${forceOcr}&parser=${parser}`, { method: "POST", body });
      setDocumentId(data.document_id); setStatus(`Ready — ${data.chunks} passages indexed from ${file.name}.`);
    } catch (error) { setStatus(error instanceof Error ? error.message : "Upload failed."); }
    finally { setBusy(null); }
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    if (!documentId || !question.trim() || busy) return;
    setBusy("chat"); setAnswer("Finding relevant passages and preparing your answer…"); setSources([]);
    try {
      const data = await request<{ answer: string; sources: Source[] }>("/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: documentId, question: question.trim() }),
      });
      setAnswer(data.answer); setSources(data.sources);
    } catch (error) { setAnswer(error instanceof Error ? error.message : "Unable to answer."); }
    finally { setBusy(null); }
  }

  return <main>
    <header><span className="eyebrow">YOUR DOCUMENT WORKSPACE</span><h1>Ask your PDF</h1><p>Find answers in your documents, with passages you can check.</p></header>
    <section aria-labelledby="upload-title"><div className="section-heading"><span className="step">1</span><h2 id="upload-title">Add a document</h2></div>
      <form onSubmit={upload}>
        <label className="file-label" htmlFor="pdf">Choose a PDF <span>Up to 50 MB</span></label>
        <input id="pdf" type="file" accept="application/pdf,.pdf" disabled={!!busy} onChange={e => { setFile(e.target.files?.[0] || null); setDocumentId(null); setAnswer(""); setSources([]); setStatus("Upload your selected PDF to index it."); }} />
        <label className="parser-label" htmlFor="parser">PDF extraction</label>
        <select id="parser" value={parser} disabled={!!busy} onChange={e => { const value = e.target.value as "docling" | "pymupdf"; setParser(value); if (value === "pymupdf") setForceOcr(false); }}>
          <option value="docling">Docling — scanned PDFs, tables and layouts</option>
          <option value="pymupdf">PyMuPDF — fast extraction of existing text</option>
        </select>
        <label className="checkbox"><input type="checkbox" checked={forceOcr} disabled={!!busy || parser === "pymupdf"} onChange={e => setForceOcr(e.target.checked)} />Force OCR for scanned pages or unreadable text</label>
        <button disabled={!file || !!busy}>{busy === "upload" ? "Indexing…" : "Upload and index"}</button>
        <p className="status" role="status">{status}</p>
      </form>
    </section>
    <section aria-labelledby="question-title"><div className="section-heading"><span className="step">2</span><h2 id="question-title">Ask a question</h2></div>
      <form onSubmit={ask}><label htmlFor="question">What would you like to know?</label><textarea id="question" value={question} onChange={e => setQuestion(e.target.value)} rows={3} maxLength={4000} placeholder="What are the main findings?" disabled={!!busy} /><button disabled={!documentId || !question.trim() || !!busy}>{busy === "chat" ? "Preparing answer…" : "Ask document"}</button></form>
      {answer && <div className="answer" aria-live="polite">{answer}</div>}
      {!!sources.length && <div className="sources"><h3>Source passages</h3>{sources.map(source => <details key={source.source}><summary><span className="citation">[{source.source}]</span> {source.filename} · Page {source.page ?? "unknown"}</summary><p>{source.text}</p></details>)}</div>}
    </section>
    <footer>Your question and relevant passages are sent to your configured answer provider.</footer>
  </main>;
}
