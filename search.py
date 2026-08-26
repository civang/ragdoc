"""
ragdoc — search chunks by meaning.

Now we connect the pieces. Given a question, we:
  1. embed the question into a vector (same model as the chunks),
  2. compare it against every chunk's vector with cosine similarity,
  3. return the top few chunks - the parts of the document most likely to
     contain the answer.

This is the "Retrieval" in Retrieval-Augmented Generation. ask.py then hands
these top chunks to Claude to write the actual answer.

Run it directly to search a document interactively:

    py search.py path/to/your/document.pdf
"""

import hashlib
import os
import pickle
import sys

import numpy as np

from chunk import chunk_text, load_text
from embed import cosine_similarity, embed_texts

# Where cached indexes live. Gitignored - it's build output, not source.
CACHE_DIR = ".ragdoc_cache"


# Each indexed chunk is a record: {"text": <the chunk>, "source": <filename>}.
# Carrying the source lets answers say WHICH document a passage came from,
# which matters once we index more than one file.

SUPPORTED_EXTS = {".pdf", ".txt", ".md"}
_CACHE_VERSION = 3  # bumped when the cache format changes


def _cache_path(doc_path: str) -> str:
    """A unique cache filename for a document, based on its absolute path."""
    key = os.path.abspath(doc_path)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{digest}.pkl")


def _index_one_file(
    path: str, use_cache: bool, source_name: str | None = None
) -> tuple[list[dict], np.ndarray]:
    """Chunk + embed a single document, reusing the on-disk cache when possible.

    `source_name` is the label shown in answers. When indexing a folder it is
    the path relative to that folder, so nested files with the same basename
    (a/notes.txt vs b/notes.txt) stay distinguishable.
    """
    source = source_name or os.path.basename(path)
    cache_file = _cache_path(path)
    mtime = os.path.getmtime(path)

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        if (
            cached.get("version") == _CACHE_VERSION
            and cached.get("mtime") == mtime
            and cached.get("source") == source
        ):
            print(f"(loaded from cache: {source})")
            return cached["chunks"], cached["vectors"]

    text = load_text(path)
    chunks = [{"text": t, "source": source} for t in chunk_text(text)]
    vectors = embed_texts([c["text"] for c in chunks])

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(
                {"version": _CACHE_VERSION, "mtime": mtime, "source": source,
                 "chunks": chunks, "vectors": vectors},
                f,
            )

    return chunks, vectors


def build_index(path: str, use_cache: bool = True) -> tuple[list[dict], np.ndarray]:
    """Build a search index from a single document OR a whole folder.

    If `path` is a file, index it. If it's a folder, index every supported
    file inside it (.pdf/.txt/.md) into ONE combined index, so a question can
    be answered from any of the documents.

    Returns (chunks, vectors): row i of `vectors` is the embedding of
    `chunks[i]["text"]`, and `chunks[i]["source"]` names the file it came from.
    """
    if os.path.isdir(path):
        files = _find_documents(path)
        if not files:
            raise ValueError(f"No supported documents (.pdf/.txt/.md) found in: {path}")
        print(f"Indexing {len(files)} documents from {path}/")

        all_chunks: list[dict] = []
        all_vectors: list[np.ndarray] = []
        skipped: list[str] = []

        for file_path in files:
            # Label nested files by their path relative to the folder, so
            # a/notes.txt and b/notes.txt don't both show up as "notes.txt".
            label = os.path.relpath(file_path, path).replace(os.sep, "/")

            # One unreadable file shouldn't take down the whole index. Report
            # it and carry on with the rest.
            try:
                chunks, vectors = _index_one_file(file_path, use_cache, label)
            except Exception as e:
                skipped.append(f"{label} ({type(e).__name__})")
                continue
            all_chunks.extend(chunks)
            all_vectors.append(vectors)

        if skipped:
            print(f"  skipped {len(skipped)} unreadable file(s): {', '.join(skipped)}")
        if not all_chunks:
            raise ValueError(f"No readable content found in: {path}")

        return all_chunks, np.vstack(all_vectors)

    return _index_one_file(path, use_cache)


def _find_documents(folder: str) -> list[str]:
    """Every supported document under `folder`, including subdirectories.

    Pointing at a docs/ tree should index the whole tree - silently ignoring
    nested files gives partial answers with no indication anything was missed.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        # Skip hidden and cache directories (.git, .ragdoc_cache, ...).
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def search(
    query: str,
    chunks: list[dict],
    vectors: np.ndarray,
    top_k: int = 3,
) -> list[tuple[float, dict]]:
    """Return the top_k chunks most similar in meaning to the query.

    Each result is (similarity_score, chunk_record), best match first, where
    chunk_record is {"text": ..., "source": ...}.
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
            text = chunk["text"]
            preview = text[:200].replace("\n", " ")
            print(f"  #{rank}  (similarity {score:.3f} | {chunk['source']})")
            print(f"      {preview}{'...' if len(text) > 200 else ''}\n")


if __name__ == "__main__":
    main()
