"""
ragdoc — baseline: chat with a document (no retrieval).

Load a .txt, .md, or .pdf file and ask Claude questions about it in a loop.
This version sends the whole document to the model each turn (no embeddings yet
— see ask.py for that). Kept as the simplest thing that actually works, and
as a before/after contrast with the real RAG pipeline.

Usage:
    py ragdoc.py path/to/your/document.pdf

Requires an Anthropic API key. See README.md for setup.
"""

import base64
import os
import sys

import anthropic

MODEL = "claude-opus-4-8"          # swap to "claude-haiku-4-5" for cheaper testing
MAX_TOKENS = 2048


def load_document_block(path: str) -> dict:
    """Turn a file into a content block Claude can read.

    .pdf  -> a base64 'document' block (Claude reads the PDF natively)
    .txt/.md and other text -> a plain 'text' block
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": data,
            },
            # cache_control lets Claude reuse the document across questions,
            # so you only pay full price to read it the first time.
            "cache_control": {"type": "ephemeral"},
        }

    # Everything else: read as text.
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return {
        "type": "text",
        "text": f"Here is the document to answer questions about:\n\n{text}",
        "cache_control": {"type": "ephemeral"},
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: py ragdoc.py <path-to-document>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    document_block = load_document_block(path)

    print(f"\nLoaded: {path}")
    print("Ask a question about it. Type 'exit' or press Ctrl+C to quit.\n")

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

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "You answer questions using ONLY the provided document. "
                "If the answer isn't in the document, say so plainly. "
                "Quote or point to the relevant part when you can."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        document_block,
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )

        answer = "".join(b.text for b in response.content if b.type == "text")
        print(f"\nragdoc: {answer}\n")


if __name__ == "__main__":
    main()
