"""
ragdoc — Day 2: split a document into chunks.

Why chunk? On Day 1 we sent the whole document to Claude every time. That works
for small files, but it gets slow and expensive as documents grow. The RAG idea
is: break the document into small pieces ("chunks"), and later (Week 1) only send
the few pieces relevant to each question.

Today we just build and eyeball the chunker. Run it directly to see it work:

    py chunk.py path/to/your/document.pdf
"""

import os
import sys


def load_text(path: str) -> str:
    """Read a file's text. Supports .pdf (via pypdf) and any text file."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print("PDF support needs pypdf. Run: py -m pip install -r requirements.txt")
            sys.exit(1)
        reader = PdfReader(path)
        # Join the text of every page with blank lines between them.
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks.

    chunk_size: how many characters per chunk (roughly a few paragraphs).
    overlap:    how many characters each chunk repeats from the one before it.

    Why overlap? A sentence that answers a question might sit right on the border
    between two chunks. Repeating a little text at each boundary means the full
    thought lands intact in at least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap  # how far we advance each time

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: py chunk.py <path-to-document>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    text = load_text(path)
    chunks = chunk_text(text)

    print(f"\nLoaded: {path}")
    print(f"Document length: {len(text):,} characters")
    print(f"Split into: {len(chunks)} chunks\n")

    # Preview the first two chunks so you can see the overlap in action.
    for i, chunk in enumerate(chunks[:2]):
        preview = chunk[:200].replace("\n", " ")
        print(f"--- chunk {i} ({len(chunk)} chars) ---")
        print(preview + ("..." if len(chunk) > 200 else ""))
        print()

    if len(chunks) > 2:
        print(f"...and {len(chunks) - 2} more.")


if __name__ == "__main__":
    main()
