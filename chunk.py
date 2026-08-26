"""
ragdoc — split a document into chunks.

Why chunk? Sending a whole document to the model works for small files, but it
gets slow and expensive as documents grow. The RAG idea is: break the document
into small pieces ("chunks"), then send only the few pieces relevant to each
question.

Chunks are split on sentence boundaries, because a fragment that starts mid-word
embeds worse than a complete thought. Run it directly to inspect the output:

    py chunk.py path/to/your/document.pdf
"""

import os
import re
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

    # Real-world folders contain files saved in older encodings. Try UTF-8
    # first, then fall back to latin-1, which maps every byte to a character
    # and so never raises. Better to index slightly mangled text than to fail.
    raw = open(path, "rb").read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


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

    # Pack whole sentences into chunks instead of slicing at a fixed character
    # count. A chunk that starts mid-word ("...ergy needed to power") embeds
    # poorly, because a fragment carries muddier meaning than a full thought.
    sentences = split_sentences(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        # A single sentence longer than chunk_size can't be packed - split it
        # on character boundaries as a fallback so we never drop text.
        if len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(_hard_split(sentence, chunk_size, overlap))
            continue

        # +1 accounts for the space we join sentences with.
        if current and current_len + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # Start the next chunk with trailing sentences from this one, so a
            # thought spanning the boundary stays intact in at least one chunk.
            current = _tail_sentences(current, overlap)
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping paragraph breaks as boundaries.

    Deliberately simple: a sentence ends at . ! or ? followed by whitespace.
    Good enough for prose, and it avoids pulling in a heavyweight NLP library.
    """
    sentences: list[str] = []

    # Paragraphs are natural boundaries - never merge across a blank line.
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Split after sentence-ending punctuation followed by whitespace.
        parts = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences


def _tail_sentences(sentences: list[str], overlap: int) -> list[str]:
    """Return the trailing sentences that fit within `overlap` characters."""
    if overlap <= 0:
        return []

    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > overlap:
            break
        tail.insert(0, sentence)
        total += len(sentence) + 1

    return tail


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback character-based split, for text with no sentence breaks."""
    pieces: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            pieces.append(piece)
        start += step

    return pieces


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
