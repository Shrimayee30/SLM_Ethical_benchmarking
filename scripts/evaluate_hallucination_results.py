#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethical_benchmarking.hallucination import read_jsonl, score_item, summarize_scores, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hallucination benchmark responses.")
    parser.add_argument("--dataset", default="data/hallucination/eval.jsonl")
    parser.add_argument("--results", required=True)
    parser.add_argument("--scores-output", default=None)
    args = parser.parse_args()

    items = {row["id"]: row for row in read_jsonl(ROOT / args.dataset)}
    results = read_jsonl(ROOT / args.results)

    scores_by_model = defaultdict(list)
    score_rows = []
    for result in results:
        item = items[result["item_id"]]
        score = score_item(item, result["response"])
        scores_by_model[result["model"]].append(score)
        score_rows.append(
            {
                "model": result["model"],
                "item_id": score.item_id,
                "task_type": score.task_type,
                "label": score.label,
                "hallucination_risk": score.hallucination_risk,
                "capability_correct": score.capability_correct,
                "refused": score.refused,
                "reason": score.reason,
            }
        )

    report = {model: summarize_scores(scores) for model, scores in sorted(scores_by_model.items())}
    print(json.dumps(report, indent=2))

    if args.scores_output:
        write_jsonl(ROOT / args.scores_output, score_rows)


if __name__ == "__main__":
    main()
