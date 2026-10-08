"use client";

import { useState } from "react";

type Source = { source: number; filename: string; page: number | null; text: string; score: number };
type Indexed = { document_id: string; filename: string; chunks: number; replaced: boolean };
type UploadResponse = { documents?: Indexed[]; document_id: string; filename?: string; chunks: number; replaced?: boolean };

async function request<T>(path: string, options: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(typeof data?.detail === "string" ? data.detail : `Request failed (${response.status}). Please try again.`);
  if (!data) throw new Error("The server returned an unexpected response.");
  return data as T;
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [forceOcr, setForceOcr] = useState(false);
  const [parser, setParser] = useState<"docling" | "pymupdf">("docling");
  const [documents, setDocuments] = useState<Indexed[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("Upload one or more PDFs to get started.");
  const [busy, setBusy] = useState<"upload" | "chat" | null>(null);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);

  async function upload(event: React.FormEvent) {
    event.preventDefault();
    if (!files.length || busy) return;
    if (files.some(f => f.size > 50 * 1024 * 1024)) { setStatus("Every PDF must be smaller than 50 MB."); return; }
    setBusy("upload"); setAnswer(""); setSources([]);
    setStatus(`Indexing ${files.length} PDF${files.length > 1 ? "s" : ""}. This may take a few minutes…`);
    try {
      const body = new FormData();
      for (const f of files) body.append("file", f);
      const data = await request<UploadResponse>(`/documents?force_ocr=${forceOcr}&parser=${parser}`, { method: "POST", body });
      const indexed: Indexed[] = data.documents ?? [{ document_id: data.document_id, filename: data.filename ?? files[0].name, chunks: data.chunks, replaced: data.replaced ?? false }];
      setDocuments(prev => {
        const byId = new Map(prev.map(d => [d.document_id, d]));
        for (const doc of indexed) byId.set(doc.document_id, doc);
        return [...byId.values()];
      });
      setSelected(prev => new Set([...prev, ...indexed.map(d => d.document_id)]));
      setFiles([]);
      const failed = (data as { failures?: { filename: string }[] }).failures ?? [];
      setStatus(`Indexed ${indexed.length} PDF${indexed.length > 1 ? "s" : ""}${failed.length ? `, ${failed.length} failed (${failed.map(f => f.filename).join(", ")})` : ""}. Ask about any selection below.`);
    } catch (error) { setStatus(error instanceof Error ? error.message : "Upload failed."); }
    finally { setBusy(null); }
  }

  function toggle(id: string) {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const ids = documents.filter(d => selected.has(d.document_id)).map(d => d.document_id);
    if (!ids.length || !question.trim() || busy) return;
    setBusy("chat"); setAnswer("Finding relevant passages and preparing your answer…"); setSources([]);
    try {
      const data = await request<{ answer: string; sources: Source[] }>("/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_ids: ids, question: question.trim() }),
      });
      setAnswer(data.answer); setSources(data.sources);
    } catch (error) { setAnswer(error instanceof Error ? error.message : "Unable to answer."); }
    finally { setBusy(null); }
  }

  const selectedCount = documents.filter(d => selected.has(d.document_id)).length;

  return <main>
    <header><span className="eyebrow">YOUR DOCUMENT WORKSPACE</span><h1>Ask your PDFs</h1><p>Find answers in your documents, with passages you can check.</p></header>
    <section aria-labelledby="upload-title"><div className="section-heading"><span className="step">1</span><h2 id="upload-title">Add documents</h2></div>
      <form onSubmit={upload}>
        <label className="file-label" htmlFor="pdf">Choose PDFs <span>Multiple allowed, up to 50 MB each</span></label>
        <input id="pdf" type="file" accept="application/pdf,.pdf" multiple disabled={!!busy} onChange={e => { setFiles(Array.from(e.target.files ?? [])); setAnswer(""); setSources([]); setStatus(files.length ? "" : "Select PDFs and upload to index them."); }} />
        <p className="status">{files.length ? `${files.length} file${files.length > 1 ? "s" : ""} selected: ${files.map(f => f.name).join(", ")}` : ""}</p>
        <label className="parser-label" htmlFor="parser">PDF extraction</label>
        <select id="parser" value={parser} disabled={!!busy} onChange={e => { const value = e.target.value as "docling" | "pymupdf"; setParser(value); if (value === "pymupdf") setForceOcr(false); }}>
          <option value="docling">Docling — scanned PDFs, tables and layouts</option>
          <option value="pymupdf">PyMuPDF — fast extraction of existing text</option>
        </select>
        <label className="checkbox"><input type="checkbox" checked={forceOcr} disabled={!!busy || parser === "pymupdf"} onChange={e => setForceOcr(e.target.checked)} />Force OCR for scanned pages or unreadable text</label>
        <button disabled={!files.length || !!busy}>{busy === "upload" ? "Indexing…" : "Upload and index"}</button>
        <p className="status" role="status">{status}</p>
      </form>
    </section>
    {!!documents.length && <section aria-labelledby="library-title"><div className="section-heading"><span className="step">•</span><h2 id="library-title">Document library ({documents.length})</h2></div>
      {documents.map(doc => <label className="checkbox" key={doc.document_id}>
        <input type="checkbox" checked={selected.has(doc.document_id)} disabled={!!busy} onChange={() => toggle(doc.document_id)} />
        {doc.filename} — {doc.chunks} passages{doc.replaced ? " (re-indexed)" : ""}
      </label>)}
    </section>}
    <section aria-labelledby="question-title"><div className="section-heading"><span className="step">2</span><h2 id="question-title">Ask a question</h2></div>
      <form onSubmit={ask}><label htmlFor="question">What would you like to know? (searches {selectedCount} selected document{selectedCount === 1 ? "" : "s"})</label><textarea id="question" value={question} onChange={e => setQuestion(e.target.value)} rows={3} maxLength={4000} placeholder="What are the main findings?" disabled={!!busy} /><button disabled={!selectedCount || !question.trim() || !!busy}>{busy === "chat" ? "Preparing answer…" : "Ask documents"}</button></form>
      {answer && <div className="answer" aria-live="polite">{answer}</div>}
      {!!sources.length && <div className="sources"><h3>Source passages</h3>{sources.map(source => <details key={source.source}><summary><span className="citation">[{source.source}]</span> {source.filename} · Page {source.page ?? "unknown"}</summary><p>{source.text}</p></details>)}</div>}
    </section>
    <footer>Your question and relevant passages are sent to your configured answer provider.</footer>
  </main>;
}
