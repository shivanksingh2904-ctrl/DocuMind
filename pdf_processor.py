"""PDF reading and sentence-aware chunking for DocuMind."""
import hashlib
import os
import re

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
TARGET_CHARS = int(os.getenv("CHUNK_CHARS", 900))  # approx. size of one chunk
OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", 1))


def read_pdf(path: str) -> list[tuple[int, str]]:
    """Return [(page_number, cleaned_text)] for pages that contain text."""
    from pypdf import PdfReader

    pages = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((number, text))
    return pages


def chunk_pages(pages, doc_name, target=TARGET_CHARS, overlap=OVERLAP_SENTENCES):
    """Group whole sentences into chunks of ~`target` characters.

    Unlike fixed word windows, chunks never cut a sentence in half, and each
    chunk remembers the page range it came from for citations.
    """
    units = [(s.strip(), pg) for pg, text in pages for s in SENTENCE.split(text) if s.strip()]
    chunks, current, size, fresh = [], [], 0, 0

    def flush():
        text = " ".join(s for s, _ in current)
        pgs = [p for _, p in current]
        chunks.append({
            "id": hashlib.md5(f"{doc_name}|{pgs[0]}|{text}".encode()).hexdigest(),
            "doc": doc_name, "page": pgs[0], "page_end": pgs[-1], "text": text,
        })

    for unit in units:
        if current and size + len(unit[0]) > target and fresh:
            flush()
            current = current[-overlap:] if overlap else []
            size, fresh = sum(len(s) for s, _ in current), 0
        current.append(unit)
        size += len(unit[0])
        fresh += 1
    if current and fresh:
        flush()
    return chunks


def process_pdf(path: str, doc_name: str | None = None):
    """Return (chunks, info) for one PDF."""
    doc_name = doc_name or os.path.basename(path)
    pages = read_pdf(path)
    chunks = chunk_pages(pages, doc_name)
    words = sum(len(t.split()) for _, t in pages)
    return chunks, {"pages": len(pages), "words": words}
