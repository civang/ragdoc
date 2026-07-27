"""
ragdoc — Day 4: search chunks by meaning.

Now we connect the pieces. Given a question, we:
  1. embed the question into a vector (same model as the chunks),
  2. compare it against every chunk's vector with cosine similarity,
  3. return the top few chunks - the parts of the document most likely to
     contain the answer.

This is the "Retrieval" in Retrieval-Augmented Generation. Tomorrow (Day 5) we
hand these top chunks to Claude to write the actual answer.

Run it directly to search a document interactively:

    py search.py path/to/your/document.pdf
"""

import sys

import numpy as np

from chunk import chunk_text, load_text
from embed import cosine_similarity, embed_texts


def build_index(path: str) -> tuple[list[str], np.ndarray]:
    """Load a document, chunk it, and embed every chunk.

    The returned (chunks, vectors) pair is our tiny search index: row i of
    `vectors` is the embedding of `chunks[i]`.
    """
    text = load_text(path)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    return chunks, vectors


def search(
    query: str,
    chunks: list[str],
    vectors: np.ndarray,
    top_k: int = 3,
) -> list[tuple[float, str]]:
    """Return the top_k chunks most similar in meaning to the query.

    Each result is (similarity_score, chunk_text), best match first.
    """
    query_vec = embed_texts([query])[0]

    # Score every chunk against the query.
    scored = [
        (cosine_similarity(query_vec, vectors[i]), chunks[i])
        for i in range(len(chunks))
    ]

    # Sort by score, highest first, and keep the best few.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: py search.py <path-to-document>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"\nBuilding search index for: {path}")
    chunks, vectors = build_index(path)
    print(f"Indexed {len(chunks)} chunks.")
    print("Type a question to find the most relevant chunks. 'exit' to quit.\n")

    while True:
        try:
            query = input("Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        results = search(query, chunks, vectors, top_k=3)
        print()
        for rank, (score, chunk) in enumerate(results, start=1):
            preview = chunk[:200].replace("\n", " ")
            print(f"  #{rank}  (similarity {score:.3f})")
            print(f"      {preview}{'...' if len(chunk) > 200 else ''}\n")


if __name__ == "__main__":
    main()
