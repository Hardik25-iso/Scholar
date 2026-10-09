"""Compare two eval runs question by question.

    python scripts/eval_diff.py BASELINE.json CANDIDATE.json [--max-drop 1.0]

Both files come from `python -m backend.eval --json-out`. Prints every question
whose first-hit rank changed and the summary deltas, and exits non-zero if
hit@k fell by more than --max-drop points OR any question flipped from hit to
miss. The second condition is the real bar on a ~50-question set, where one
question is ~2 points: "within 1 point" there means "no question lost".
"""
import argparse
import json
import sys


def main() -> int:
    p = argparse.ArgumentParser(prog="python scripts/eval_diff.py")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--max-drop", type=float, default=1.0, help="allowed hit@k drop, in points")
    args = p.parse_args()

    base = json.load(open(args.baseline, encoding="utf-8"))
    cand = json.load(open(args.candidate, encoding="utf-8"))

    lost, gained, moved = [], [], []
    for qid, before in base["ranks"].items():
        after = cand["ranks"].get(qid)
        if before == after:
            continue
        if before is not None and after is None:
            lost.append((qid, before, after))
        elif before is None and after is not None:
            gained.append((qid, before, after))
        else:
            moved.append((qid, before, after))

    for label, rows in (("LOST (hit -> miss)", lost), ("gained (miss -> hit)", gained),
                        ("rank changed", moved)):
        if rows:
            print(f"{label}:")
            for qid, before, after in rows:
                print(f"  {qid:<12} {before!s:>4} -> {after!s}")

    drop = (base["hit_at_k"] - cand["hit_at_k"]) * 100
    print(f"\nhit@{base['k']}  {base['hit_at_k']:.1%} -> {cand['hit_at_k']:.1%}  ({-drop:+.1f} pts)")
    print(f"MRR    {base['mrr']:.3f} -> {cand['mrr']:.3f}")
    print(f"hit@1  {base['hit_at_1']:.1%} -> {cand['hit_at_1']:.1%}")

    ok = drop <= args.max_drop and not lost
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
