"""
PDF Ingestion Engine
====================
Extracts text from PDFs with rich metadata.
Now uses Content Hashing (SHA-256) to identify unique documents.
"""

import re
import hashlib
import uuid
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Chunk:
    """A self-contained piece of text with full provenance metadata."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        doc_name: str,
        page_number: int,         # 1-indexed
        paragraph_index: int,
        sentence_start: int,      # char offset within paragraph
        sentence_end: int,
        text: str,
        section_heading: Optional[str],
        bbox: Optional[tuple],    # (x0, y0, x1, y1) on page
        chunk_index: int,         # position in document chunk list
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.page_number = page_number
        self.paragraph_index = paragraph_index
        self.sentence_start = sentence_start
        self.sentence_end = sentence_end
        self.text = text
        self.section_heading = section_heading
        self.bbox = bbox
        self.chunk_index = chunk_index

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "paragraph_index": self.paragraph_index,
            "sentence_start": self.sentence_start,
            "sentence_end": self.sentence_end,
            "text": self.text,
            "section_heading": self.section_heading,
            "bbox": list(self.bbox) if self.bbox else None,
            "chunk_index": self.chunk_index,
        }


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""
    parts = _SENT_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Heading detector
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r'^(?:\d+\.?\s+[A-Z]|[A-Z][A-Z\s]{3,}$|Abstract|Introduction|'
    r'Related Work|Methodology|Experiments?|Results?|Discussion|'
    r'Conclusion|References?|Appendix)',
    re.MULTILINE
)

def _detect_heading(text: str) -> bool:
    """Return True if this block looks like a section heading."""
    text = text.strip()
    if len(text) > 120:
        return False
    if _HEADING_RE.match(text):
        return True
    if text.isupper() and 3 <= len(text.split()) <= 8:
        return True
    return False


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def _make_chunk_id(doc_id: str, page: int, para: int, start: int) -> str:
    # Adding a short random uuid makes it impossible to have duplicates
    salt = uuid.uuid4().hex[:8]
    raw = f"{doc_id}:{page}:{para}:{start}:{salt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def calculate_file_hash(path: str | Path) -> str:
    """Generate a SHA-256 hash of the file contents for unique identification."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in blocks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:32]


def ingest_pdf(path: str | Path) -> list[Chunk]:
    """
    Parse a PDF and return a list of Chunk objects.
    Uses file content hash as the doc_id to detect duplicates.
    """
    path = Path(path)
    doc_id = calculate_file_hash(path)  # CONTENT-BASED ID
    doc_name = path.stem

    chunks: list[Chunk] = []
    chunk_index = 0
    current_heading: Optional[str] = None

    doc = fitz.open(str(path))

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks", sort=True)

        for para_idx, block in enumerate(blocks):
            x0, y0, x1, y1, raw_text, _, block_type = block
            if block_type != 0:
                continue

            text = raw_text.strip()
            if not text or len(text) < 10:
                continue

            if _detect_heading(text):
                current_heading = text[:80]
                continue

            sentences = _split_sentences(text)
            if not sentences:
                continue

            buffer = ""
            buf_start = 0
            buf_sentences: list[str] = []

            for sent in sentences:
                candidate = (buffer + " " + sent).strip() if buffer else sent

                if len(candidate) <= CHUNK_SIZE:
                    buffer = candidate
                    buf_sentences.append(sent)
                else:
                    if buffer:
                        cid = _make_chunk_id(doc_id, page_num, para_idx, buf_start)
                        chunks.append(Chunk(
                            chunk_id=cid,
                            doc_id=doc_id,
                            doc_name=doc_name,
                            page_number=page_num,
                            paragraph_index=para_idx,
                            sentence_start=buf_start,
                            sentence_end=buf_start + len(buffer),
                            text=buffer,
                            section_heading=current_heading,
                            bbox=(x0, y0, x1, y1),
                            chunk_index=chunk_index,
                        ))
                        chunk_index += 1

                        overlap_text = buffer[-CHUNK_OVERLAP:] if len(buffer) > CHUNK_OVERLAP else buffer
                        buf_start = buf_start + len(buffer) - len(overlap_text)
                        buffer = (overlap_text + " " + sent).strip()
                        buf_sentences = [sent]
                    else:
                        buffer = sent
                        buf_sentences = [sent]

            if buffer:
                cid = _make_chunk_id(doc_id, page_num, para_idx, buf_start)
                chunks.append(Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page_number=page_num,
                    paragraph_index=para_idx,
                    sentence_start=buf_start,
                    sentence_end=buf_start + len(buffer),
                    text=buffer,
                    section_heading=current_heading,
                    bbox=(x0, y0, x1, y1),
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

    doc.close()
    return chunks


def get_doc_id(path: str | Path) -> str:
    """Utility to get doc_id without full ingestion."""
    return calculate_file_hash(path)
