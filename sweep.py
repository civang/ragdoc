"""
ragdoc — parameter sweep.

Chunk size, overlap and top_k are guesses until you measure them. This runs
the eval suite across a grid of settings and prints a table, so the defaults
in the code can be chosen from evidence rather than intuition.

    py sweep.py              # chunk_size x overlap, at top_k=3
    py sweep.py --top-k      # top_k sweep at the default chunk settings

Reading the table:
  answer recall is the metric to optimise - if the answer text never lands in
  the retrieved chunks, the model cannot answer correctly no matter how good
  the prompt is. Hit@1 and MRR describe ranking quality on top of that.

Note that bigger chunks trivially raise recall by dragging in more text per
result, so the table also reports the characters sent to the model. The best
setting is the one that keeps recall high while sending less.
"""

import argparse

from evals import evaluate

CHUNK_SIZES = [300, 500, 800, 1200, 1600]
OVERLAPS = [0, 100, 200]
TOP_KS = [1, 2, 3, 5, 8]


def context_cost(chunk_size: int, top_k: int) -> int:
    """Roughly how many characters get sent to the model per question."""
    return chunk_size * top_k


def run_grid() -> list[dict]:
    """Sweep chunk_size x overlap at a fixed top_k."""
    rows = []
    print(f"\n{'chunk':>6} {'overlap':>8} {'chunks':>7} {'recall':>8} "
          f"{'Hit@1':>7} {'MRR':>7} {'ctx chars':>10}")
    print("-" * 60)

    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            if overlap >= chunk_size:
                continue
            m = evaluate(chunk_size=chunk_size, overlap=overlap, top_k=3)
            m["context_chars"] = context_cost(chunk_size, 3)
            rows.append(m)
            print(f"{chunk_size:>6} {overlap:>8} {m['chunks_indexed']:>7} "
                  f"{m['answer_recall']:>7.0%} {m['hit_at_1']:>7.0%} "
                  f"{m['mrr']:>7.3f} {m['context_chars']:>10,}")
    return rows


def run_top_k() -> list[dict]:
    """Sweep top_k at the default chunk settings."""
    rows = []
    print(f"\n{'top_k':>6} {'recall':>8} {'Hit@1':>7} {'MRR':>7} {'ctx chars':>10}")
    print("-" * 44)

    for top_k in TOP_KS:
        m = evaluate(chunk_size=800, overlap=100, top_k=top_k)
        m["context_chars"] = context_cost(800, top_k)
        rows.append(m)
        print(f"{top_k:>6} {m['answer_recall']:>7.0%} {m['hit_at_1']:>7.0%} "
              f"{m['mrr']:>7.3f} {m['context_chars']:>10,}")
    return rows


def summarise(rows: list[dict]) -> None:
    """Report the best setting, preferring cheaper context on a tie."""
    best_recall = max(r["answer_recall"] for r in rows)
    tied = [r for r in rows if r["answer_recall"] == best_recall]
    winner = min(tied, key=lambda r: (r["context_chars"], -r["mrr"]))

    print("\n" + "=" * 60)
    print(f"Best answer recall: {best_recall:.0%} "
          f"({len(tied)} setting(s) tied)")
    print("Cheapest setting achieving it:")
    print(f"  chunk_size={winner['chunk_size']}  overlap={winner['overlap']}  "
          f"top_k={winner['top_k']}")
    print(f"  Hit@1 {winner['hit_at_1']:.0%} | MRR {winner['mrr']:.3f} | "
          f"~{winner['context_chars']:,} chars of context per question")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep retrieval parameters.")
    parser.add_argument("--top-k", action="store_true",
                        help="sweep top_k instead of chunk_size x overlap")
    args = parser.parse_args()

    print("Running the eval suite across settings. This re-embeds the corpus "
          "for each row, so it takes a moment.")

    rows = run_top_k() if args.top_k else run_grid()
    summarise(rows)


if __name__ == "__main__":
    main()
