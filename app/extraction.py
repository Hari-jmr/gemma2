from pathlib import Path
from typing import Literal

Parser = Literal["docling", "pymupdf"]
ROOT = Path(__file__).resolve().parent.parent


def chunk_text(text, size=1600, overlap=200):
    text = text.strip()
    return [text[i:i + size] for i in range(0, len(text), size - overlap)] if text else []


def extract_pdf(path, force_ocr=False, parser: Parser = "docling"):
    if parser == "pymupdf":
        if force_ocr:
            raise ValueError("Choose Docling to use OCR. PyMuPDF mode extracts existing PDF text only.")
        import pymupdf
        with pymupdf.open(path) as document:
            return [{"page": n + 1, "text": chunk} for n, page in enumerate(document)
                    for chunk in chunk_text(page.get_text(sort=True))]
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    options = PdfPipelineOptions(do_ocr=True, artifacts_path=ROOT / "models" / "docling")
    options.ocr_options = RapidOcrOptions(force_full_page_ocr=force_ocr)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    document = converter.convert(path).document
    pages = {}
    for item, _ in document.iterate_items():
        text = getattr(item, "text", "")
        if getattr(item, "label", None) == "table":
            text = item.export_to_markdown(doc=document)
        if not text:
            continue
        provenance = getattr(item, "prov", [])
        page = provenance[0].page_no if provenance else None
        pages.setdefault(page, []).append(text)
    return [{"page": page, "text": chunk} for page, texts in pages.items()
            for chunk in chunk_text("\n".join(texts))]
