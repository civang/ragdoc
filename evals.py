"""
ragdoc — evals: measure how good retrieval actually is.

"It seems to work" is not a metric. If you change the chunk size, swap the
embedding model, or tweak top_k, you need a number that tells you whether the
system got better or worse. That's what an eval suite is for.

This measures the RETRIEVAL step only - no API key needed, no tokens spent -
because retrieval is where most RAG systems succeed or fail. If the passage
containing the answer never gets retrieved, no amount of clever prompting
saves the answer.

Metrics reported:
  Answer recall - does any retrieved chunk actually contain the answer?
                  This is the one that matters most: it is the ceiling on how
                  often the model can possibly be right.
  Hit@1         - how often the correct document is the TOP result
  Hit@3         - how often the correct document appears in the top k
  MRR           - Mean Reciprocal Rank. Right answer 1st scores 1.0,
                  2nd -> 0.5, 3rd -> 0.33. Rewards ranking it higher.

Run it:
    py evals.py

The corpus lives in eval_corpus/ as ordinary documents, long enough that they
split into several chunks - so chunk_size and overlap genuinely affect the
result and can be tuned against these numbers (see sweep.py).
"""

import os
import sys

from chunk import chunk_text, load_text
from embed import embed_texts
from search import search

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_corpus")

# Each case: the question, the document that answers it, and a phrase that must
# appear in the retrieved text for the answer to be reachable.
#
# The questions deliberately avoid the wording of the source, so keyword
# matching would fail them and only meaning-based retrieval passes.
TEST_CASES = [
    # handbook.txt
    ("How much notice do I need to give before taking leave?",
     "handbook.txt", "two weeks"),
    ("Can I fly business class on a long trip?",
     "handbook.txt", "premium economy"),
    ("What happens to leave I never used by the end of the year?",
     "handbook.txt", "forfeited"),
    ("How fast must I report a stolen laptop?",
     "handbook.txt", "twenty-four hours"),
    ("How often am I allowed to work from home?",
     "handbook.txt", "three days per week"),

    # solar_system.txt
    ("Which planet is the hottest?",
     "solar_system.txt", "four hundred and sixty"),
    ("How thick are the rings around Saturn?",
     "solar_system.txt", "tens of metres"),
    ("Where might we find life beneath a frozen surface?",
     "solar_system.txt", "Europa"),
    ("Why does one planet have decades-long seasons?",
     "solar_system.txt", "ninety-eight degrees"),

    # sourdough.txt
    ("How long before I can slice the loaf?",
     "sourdough.txt", "two hours"),
    ("What does resting the flour and water before adding salt achieve?",
     "sourdough.txt", "autolyse"),
    ("Why did my bread come out heavy and tight?",
     "sourdough.txt", "bulk ferment was cut short"),
    ("How hot should the oven be at the start?",
     "sourdough.txt", "two hundred and fifty"),

    # photosynthesis.txt
    ("Why do leaves look green to us?",
     "photosynthesis.txt", "reflects green"),
    ("Which enzyme captures carbon dioxide into sugar?",
     "photosynthesis.txt", "rubisco"),
    ("How do desert plants avoid losing water while feeding?",
     "photosynthesis.txt", "at night"),
    ("What happens when it gets too hot for the plant?",
     "photosynthesis.txt", "denature"),

    # postgres_ops.txt
    ("Why does my table keep growing when the row count is flat?",
     "postgres_ops.txt", "dead tuple"),
    ("Which index should I use for querying inside JSON?",
     "postgres_ops.txt", "GIN"),
    ("How do I find which queries are costing the most time?",
     "postgres_ops.txt", "pg_stat_statements"),
    ("What is the risk of not waiting for replicas to confirm?",
     "postgres_ops.txt", "lose recent transactions"),
]


def load_corpus() -> dict[str, str]:
    """Read the eval documents from disk."""
    corpus = {}
    for name in sorted(os.listdir(CORPUS_DIR)):
        if name.endswith((".txt", ".md")):
            corpus[name] = load_text(os.path.join(CORPUS_DIR, name))
    return corpus


def build_test_index(chunk_size: int, overlap: int):
    """Chunk and embed the corpus at the given settings (no disk cache)."""
    chunks = []
    for name, text in load_corpus().items():
        for piece in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append({"text": piece, "source": name})

    vectors = embed_texts([c["text"] for c in chunks])
    return chunks, vectors


def evaluate(chunk_size: int = 800, overlap: int = 100, top_k: int = 3,
             verbose: bool = False) -> dict:
    """Run every test case and report retrieval metrics."""
    chunks, vectors = build_test_index(chunk_size, overlap)

    hits_at_1 = 0
    hits_at_k = 0
    answers_found = 0
    reciprocal_ranks = []
    failures = []

    for question, expected_source, answer_phrase in TEST_CASES:
        results = search(question, chunks, vectors, top_k=top_k)
        retrieved_sources = [chunk["source"] for _score, chunk in results]
        retrieved_text = " ".join(chunk["text"] for _score, chunk in results)

        # The metric that matters: is the answer actually reachable?
        answer_present = answer_phrase.lower() in retrieved_text.lower()
        if answer_present:
            answers_found += 1
        else:
            failures.append((question, expected_source, answer_phrase, retrieved_sources))

        if retrieved_sources and retrieved_sources[0] == expected_source:
            hits_at_1 += 1

        if expected_source in retrieved_sources:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / (retrieved_sources.index(expected_source) + 1))
        else:
            reciprocal_ranks.append(0.0)

        if verbose:
            mark = "OK  " if answer_present else "MISS"
            print(f"  [{mark}] {question}")
            if not answer_present:
                print(f"         wanted {answer_phrase!r} - retrieved {retrieved_sources}")

    total = len(TEST_CASES)
    return {
        "total": total,
        "answer_recall": answers_found / total,
        "hit_at_1": hits_at_1 / total,
        "hit_at_k": hits_at_k / total,
        "mrr": sum(reciprocal_ranks) / total,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
        "chunks_indexed": len(chunks),
        "failures": failures,
    }


def main() -> None:
    corpus = load_corpus()
    print("\nragdoc retrieval evals")
    print("=" * 68)
    print(f"Corpus: {len(corpus)} documents | Test cases: {len(TEST_CASES)}\n")

    metrics = evaluate(verbose=True)

    print("\n" + "=" * 68)
    print("RESULTS  "
          f"(chunk_size={metrics['chunk_size']}, overlap={metrics['overlap']}, "
          f"top_k={metrics['top_k']}, {metrics['chunks_indexed']} chunks)")
    print(f"  Answer recall : {metrics['answer_recall']:.0%}   "
          "(answer text present in retrieved chunks)")
    print(f"  Hit@1         : {metrics['hit_at_1']:.0%}   (correct doc ranked first)")
    print(f"  Hit@{metrics['top_k']}         : {metrics['hit_at_k']:.0%}   "
          "(correct doc retrieved at all)")
    print(f"  MRR           : {metrics['mrr']:.3f}  (1.0 is perfect)")

    if metrics["failures"]:
        print(f"\n  {len(metrics['failures'])} case(s) where the answer was not retrieved:")
        for question, expected, phrase, got in metrics["failures"]:
            print(f"    - {question}")
            print(f"      needed {phrase!r} from {expected}, retrieved {got}")

    # Regression gates. If a change to chunking, embeddings or top_k drops
    # quality below these, CI fails rather than letting it land quietly.
    gates = {
        "answer recall": (metrics["answer_recall"], 0.75),
        "Hit@1": (metrics["hit_at_1"], 0.70),
    }

    print()
    passed = True
    for label, (value, threshold) in gates.items():
        ok = value >= threshold
        passed = passed and ok
        print(f"  Gate: {label} >= {threshold:.0%} ... {'PASS' if ok else 'FAIL'} ({value:.0%})")
    print("=" * 68 + "\n")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
