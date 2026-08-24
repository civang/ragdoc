"""
ragdoc — turn chunks into embeddings (vectors).

An *embedding* is a list of numbers that captures the *meaning* of a piece of
text. Two texts about the same topic get vectors that point in a similar
direction, even if they share no words. That's what lets us "search by meaning"
instead of by exact keywords.

We use a small local model (model2vec) so there's no second API key and it runs
offline after the first download.

Run it directly to see it work — it chunks a document, embeds the chunks, and
proves the vectors capture meaning:

    py embed.py path/to/your/document.pdf
"""

import os
import sys

# On Windows, HuggingFace's model cache uses symlinks by default, which need
# admin/Developer Mode. Tell it to copy files instead so the download "just
# works" for everyone. (Must be set before model2vec/huggingface_hub load.)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

from chunk import chunk_text, load_text

# The model is loaded lazily and cached, so we only pay the load cost once.
_MODEL = None
_MODEL_NAME = "minishlab/potion-base-8M"  # small, fast, static embeddings


def get_model():
    """Load the embedding model once and reuse it."""
    global _MODEL
    if _MODEL is None:
        try:
            from model2vec import StaticModel
        except ImportError:
            print("Embeddings need model2vec. Run: py -m pip install -r requirements.txt")
            sys.exit(1)
        print(f"Loading embedding model ({_MODEL_NAME})... first run downloads ~30 MB.")
        _MODEL = StaticModel.from_pretrained(_MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Turn a list of texts into a matrix of vectors (one row per text)."""
    model = get_model()
    return model.encode(texts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How aligned two vectors are: 1.0 = same direction, 0 = unrelated.

    We divide by the vectors' lengths so only *direction* (meaning) matters,
    not magnitude.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: py embed.py <path-to-document>")
        sys.exit(1)

    path = sys.argv[1]
    text = load_text(path)
    chunks = chunk_text(text)

    vectors = embed_texts(chunks)

    print(f"\nLoaded: {path}")
    print(f"Chunks: {len(chunks)}")
    print(f"Each chunk is now a vector of {vectors.shape[1]} numbers.")
    print(f"So the whole document is a {vectors.shape[0]} x {vectors.shape[1]} matrix.\n")

    # --- Prove embeddings capture MEANING, not just matching words ---
    a, b, c = embed_texts([
        "The cat sat on the mat.",
        "A kitten rested on the rug.",   # same meaning, different words
        "Quarterly revenue grew by 12%.",  # totally different topic
    ])
    print("Meaning check (cosine similarity, higher = more alike):")
    print(f"  cat/mat  vs  kitten/rug   : {cosine_similarity(a, b):.3f}  (should be high)")
    print(f"  cat/mat  vs  revenue       : {cosine_similarity(a, c):.3f}  (should be low)")
    print("\nNotice: 'cat' and 'kitten' share NO words, yet their vectors are close.")
    print("That's meaning-based similarity - the engine behind search.py.")


if __name__ == "__main__":
    main()
