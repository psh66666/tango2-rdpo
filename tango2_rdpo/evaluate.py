"""Summarize blind listening ratings with prompt-level bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def summarize(values: dict[str, list[float]], *, seed: int) -> dict:
    prompt_scores = [sum(scores) / len(scores) for scores in values.values() if scores]
    if not prompt_scores:
        raise ValueError("no completed ratings")
    mean = sum(prompt_scores) / len(prompt_scores)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(5000):
        draw = [rng.choice(prompt_scores) for _ in prompt_scores]
        bootstrap.append(sum(draw) / len(draw))
    bootstrap.sort()
    return {
        "prompt_count": len(prompt_scores),
        "regularized_score": round(mean, 4),
        "bootstrap_95_percent_ci": [round(bootstrap[124], 4), round(bootstrap[4874], 4)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    key = json.loads(args.answer_key.read_text())
    scores = {"alignment": defaultdict(list), "naturalness": defaultdict(list)}
    seen = set()
    with args.ratings.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rater, pair_id = row["rater_id"].strip(), row["pair_id"].strip()
            if not rater or pair_id not in key:
                raise ValueError("rating has missing rater or unknown pair")
            if (rater, pair_id) in seen:
                raise ValueError(f"duplicate rating for {rater}, {pair_id}")
            seen.add((rater, pair_id))
            for question, column in (("alignment", "alignment_choice"), ("naturalness", "naturalness_choice")):
                choice = row[column].strip().upper()
                if choice not in {"L", "R", "T"}:
                    raise ValueError(f"invalid {column}: {choice}")
                improved_side = "L" if key[pair_id]["left_model"] == "regularized" else "R"
                score = 0.5 if choice == "T" else float(choice == improved_side)
                scores[question][key[pair_id]["prompt_id"]].append(score)
    result = {question: summarize(values, seed=args.seed) for question, values in scores.items()}
    result["rating_count"] = len(seen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
