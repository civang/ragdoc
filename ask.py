"""
ragdoc — Day 5: real RAG (Retrieval-Augmented Generation).

This is the whole project coming together. Instead of sending the entire
document to Claude (like Day 1's ragdoc.py does), we now:

  1. search() the document for the chunks most relevant to the question,
  2. send ONLY those chunks to Claude as context,
  3. ask Claude to answer using just that context.

Why this is better than Day 1:
  - Cheaper & faster: we send a few small chunks, not the whole document.
  - Scales: works on a 500-page manual, where sending everything is impossible.
  - Grounded: the answer is based on retrieved passages, not vague memory.

Usage:
    py ask.py path/to/your/document.pdf     # one document
    py ask.py path/to/a/folder              # every .pdf/.txt/.md inside

Requires an Anthropic API key (see README.md).
"""

import os
import sys

import anthropic

from search import build_index, search

MODEL = "claude-opus-4-8"        # swap to "claude-haiku-4-5" for cheaper testing
MAX_TOKENS = 1024
TOP_K = 3                        # how many chunks to retrieve per question


def build_context(results: list[tuple[float, dict]]) -> str:
    """Format the retrieved chunks into a labeled context block for Claude."""
    parts = []
    for i, (score, chunk) in enumerate(results, start=1):
        parts.append(
            f"[Chunk {i} | from {chunk['source']} | similarity {score:.3f}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def answer(client: anthropic.Anthropic, question: str, context: str) -> str:
    """Ask Claude to answer the question using only the retrieved context."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=(
            "You answer questions using ONLY the provided context chunks. "
            "If the answer is not in the context, say you couldn't find it in "
            "the document. Be concise and cite which chunk(s) you used."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: py ask.py <path-to-document>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    print(f"\nBuilding search index for: {path}")
    chunks, vectors = build_index(path)
    print(f"Indexed {len(chunks)} chunks. Ask away! Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        # 1. Retrieve the most relevant chunks.
        results = search(question, chunks, vectors, top_k=TOP_K)
        context = build_context(results)

        # 2 & 3. Send only those chunks to Claude and print the grounded answer.
        reply = answer(client, question, context)
        print(f"\nragdoc: {reply}")

        # 4. Show the receipts: the exact passages the answer was based on.
        #    This is grounding - the user can verify the answer against the
        #    source, and spot when the model is reaching beyond the document.
        print("\n  Sources (retrieved passages):")
        for i, (score, chunk) in enumerate(results, start=1):
            preview = " ".join(chunk["text"][:160].split())
            print(f"    [Chunk {i} | {chunk['source']} | similarity {score:.3f}] {preview}...")
        print()


if __name__ == "__main__":
    main()
