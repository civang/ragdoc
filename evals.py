"""
ragdoc — evals: measure how good retrieval actually is.

"It seems to work" is not a metric. If you change the chunk size, swap the
embedding model, or tweak top_k, you need a number that tells you whether the
system got better or worse. That's what an eval suite is for.

This measures the RETRIEVAL step only - no API key needed, no tokens spent -
because retrieval is where most RAG systems succeed or fail. If the right
passage never gets retrieved, no amount of clever prompting saves the answer.

Metrics reported:
  Hit@1  - how often the correct document is the TOP result
  Hit@3  - how often the correct document appears in the top 3
  MRR    - Mean Reciprocal Rank. If the right answer ranks 1st you score 1.0,
           2nd -> 0.5, 3rd -> 0.33, and so on. Rewards ranking it higher.

Run it:
    py evals.py
"""

import sys

from embed import embed_texts
from search import search

# ---------------------------------------------------------------------------
# The test corpus. Small, self-contained, and deliberately varied so we can
# check that questions route to the right document by MEANING, not keywords.
# ---------------------------------------------------------------------------
CORPUS = {
    "biology.txt": (
        "Photosynthesis is how plants convert sunlight into chemical energy "
        "using chlorophyll in their leaves. The process takes in carbon "
        "dioxide and water and releases oxygen as a byproduct. It is the "
        "foundation of nearly every food chain on Earth."
    ),
    "finance.txt": (
        "The company reported that revenue grew 12 percent last quarter, "
        "driven by strong software subscription sales. Operating margin "
        "improved to 23 percent. Management raised full year guidance citing "
        "continued enterprise demand."
    ),
    "geography.md": (
        "Mount Everest is the tallest mountain on Earth, standing at 8,849 "
        "meters above sea level. It sits on the border between Nepal and "
        "Tibet in the Himalayan range. The first confirmed summit was in 1953."
    ),
    "cooking.txt": (
        "To make bread you combine flour, water, yeast and salt, then knead "
        "the dough until it is smooth. Let it rise for about an hour before "
        "baking at 220 degrees Celsius until the crust is golden brown."
    ),
    "space.txt": (
        "Jupiter is the largest planet in our solar system and has at least "
        "95 known moons. Its Great Red Spot is a storm that has raged for "
        "centuries. A single Jupiter year lasts roughly 12 Earth years."
    ),
}

# Questions paired with the document that should answer them.
# Note how few of these share keywords with their target document - that's the
# point. Keyword search would fail most of these; embeddings should not.
TEST_CASES = [
    ("How do plants make their own food?", "biology.txt"),
    ("What gas do leaves release into the air?", "biology.txt"),
    ("Did the business have a good three months?", "finance.txt"),
    ("Were profits up and did the outlook improve?", "finance.txt"),
    ("How high is the world's highest peak?", "geography.md"),
    ("Which countries border the top of the Himalayas?", "geography.md"),
    ("How do I bake a loaf?", "cooking.txt"),
    ("What temperature should the oven be for dough?", "cooking.txt"),
    ("Which planet is the biggest?", "space.txt"),
    ("Tell me about the giant storm on a gas giant.", "space.txt"),
]


def build_test_index():
    """Build an in-memory index over the eval corpus (no disk cache)."""
    chunks = [{"text": text, "source": name} for name, text in CORPUS.items()]
    vectors = embed_texts([c["text"] for c in chunks])
    return chunks, vectors


def evaluate(top_k: int = 3, verbose: bool = True) -> dict:
    """Run every test case and report retrieval metrics."""
    chunks, vectors = build_test_index()

    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    failures = []

    for question, expected_source in TEST_CASES:
        results = search(question, chunks, vectors, top_k=top_k)
        retrieved = [chunk["source"] for _score, chunk in results]

        if retrieved and retrieved[0] == expected_source:
            hits_at_1 += 1

        if expected_source in retrieved:
            hits_at_k += 1
            rank = retrieved.index(expected_source) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            failures.append((question, expected_source, retrieved))

        if verbose:
            mark = "PASS" if retrieved and retrieved[0] == expected_source else "MISS"
            top_score = results[0][0] if results else 0.0
            print(f"  [{mark}] {question}")
            print(f"         expected {expected_source}, got {retrieved[0]} ({top_score:.3f})")

    total = len(TEST_CASES)
    return {
        "total": total,
        "hit_at_1": hits_at_1 / total,
        "hit_at_k": hits_at_k / total,
        "mrr": sum(reciprocal_ranks) / total,
        "top_k": top_k,
        "failures": failures,
    }


def main() -> None:
    print("\nragdoc retrieval evals")
    print("=" * 60)
    print(f"Corpus: {len(CORPUS)} documents | Test cases: {len(TEST_CASES)}\n")

    metrics = evaluate(top_k=3)

    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"  Hit@1 : {metrics['hit_at_1']:.0%}   (correct doc ranked first)")
    print(f"  Hit@3 : {metrics['hit_at_k']:.0%}   (correct doc in top 3)")
    print(f"  MRR   : {metrics['mrr']:.3f}  (1.0 is perfect)")

    if metrics["failures"]:
        print(f"\n  {len(metrics['failures'])} case(s) missed entirely:")
        for question, expected, got in metrics["failures"]:
            print(f"    - {question!r}")
            print(f"      wanted {expected}, retrieved {got}")

    # A regression gate: fail loudly if quality drops below the bar.
    # Useful in CI so a bad change to chunking or embeddings can't sneak in.
    threshold = 0.8
    passed = metrics["hit_at_1"] >= threshold
    print(f"\n  Gate: Hit@1 >= {threshold:.0%} ... {'PASS' if passed else 'FAIL'}")
    print("=" * 60 + "\n")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
