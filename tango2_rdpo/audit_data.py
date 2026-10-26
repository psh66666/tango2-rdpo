"""Reject cross-split prompt/audio leakage in a frozen Audio-Alpaca manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from tango2_rdpo.prepare_data import normalized_prompt


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--challenge-manifest", type=Path)
    args = parser.parse_args()
    rows = {
        split: read_jsonl(args.manifest_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    ids = set()
    counts = {}
    for split, items in rows.items():
        counts[split] = {"rows": len(items), "strategies": dict(Counter(x["strategy"] for x in items))}
        for row in items:
            if row["id"] in ids:
                raise ValueError(f"duplicate row id: {row['id']}")
            ids.add(row["id"])
            owners[("prompt", row["prompt_normalized"])].add(split)
            for side in ("chosen", "rejected"):
                owners[("audio", row[side]["sha256"])].add(split)
                if row[side]["sample_rate"] != 16000 or row[side]["duration_s"] <= 0:
                    raise ValueError(f"invalid audio metadata: {row['id']} {side}")
    leakage = [
        {"kind": kind, "value": value, "splits": sorted(splits)}
        for (kind, value), splits in owners.items() if len(splits) > 1
    ]
    by_id = {row["id"]: split for split, items in rows.items() for row in items}
    near_pairs = read_jsonl(args.manifest_dir / "near_pairs.jsonl")
    near_leakage = [pair for pair in near_pairs if by_id[pair["left"]] != by_id[pair["right"]]]
    challenge_overlap = []
    challenge_count = 0
    if args.challenge_manifest:
        challenges = read_jsonl(args.challenge_manifest)
        challenge_count = len(challenges)
        all_prompts = [(row["id"], row["prompt_normalized"]) for items in rows.values() for row in items]
        for challenge in challenges:
            prompt = normalized_prompt(challenge["prompt"])
            tokens = set(prompt.split())
            for identifier, known in all_prompts:
                known_tokens = set(known.split())
                overlap = len(tokens & known_tokens) / max(1, len(tokens | known_tokens))
                if overlap >= 0.75 and SequenceMatcher(None, prompt, known).ratio() >= 0.86:
                    challenge_overlap.append({"challenge": challenge["id"], "dataset": identifier})
    eval_rows = read_jsonl(args.manifest_dir / "eval_prompts.jsonl")
    test_ids = {row["id"] for row in rows["test"]}
    if not {row["id"] for row in eval_rows} <= test_ids:
        raise ValueError("evaluation prompts are not all in the frozen test split")
    report = {
        "split_counts": counts,
        "exact_cross_split_leakage": len(leakage),
        "near_prompt_cross_split_leakage": len(near_leakage),
        "eval_prompt_count": len(eval_rows),
        "challenge_count": challenge_count,
        "challenge_near_duplicates": len(challenge_overlap),
        "examples": (leakage[:10] + near_leakage[:10] + challenge_overlap[:10]),
    }
    output = args.manifest_dir / "leakage_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if leakage or near_leakage or challenge_overlap:
        raise SystemExit("prompt leakage found; do not start training")


if __name__ == "__main__":
    main()
