"""Compare matched preference diagnostics with prompt-level uncertainty."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


METRICS = (
    "preference_correct",
    "rejected_proxy_increase_rate",
    "chosen_delta",
    "rejected_delta",
    "margin",
)


def load(path: Path) -> dict:
    result = json.loads(path.read_text())
    observations = {(row["id"], row["noise_seed"]): row for row in result["observations"]}
    if len(observations) != len(result["observations"]):
        raise ValueError(f"duplicate diagnostic observation in {path}")
    result["by_key"] = observations
    return result


def interval(values: list[float], *, seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = []
    for _ in range(5000):
        sample = [rng.choice(values) for _ in values]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return [draws[124], draws[4874]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--regularized", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    runs = {name: load(path) for name, path in (
        ("baseline", args.baseline), ("regularized", args.regularized), ("ablation", args.ablation)
    )}
    baseline = runs["baseline"]
    keys = set(baseline["by_key"])
    for name, run in runs.items():
        if run["manifest_sha256"] != baseline["manifest_sha256"] or set(run["by_key"]) != keys:
            raise ValueError(f"{name}: diagnostic pairs or manifest do not match baseline")
    prompt_ids = sorted({key[0] for key in keys})
    result = {
        "manifest_sha256": baseline["manifest_sha256"],
        "prompt_count": len(prompt_ids),
        "noise_seeds": baseline["noise_seeds"],
        "models": {},
        "paired_difference_vs_baseline": {},
    }
    for name, run in runs.items():
        result["models"][name] = {metric: run["metrics_mean"][metric] for metric in METRICS}
    for name in ("regularized", "ablation"):
        difference = {}
        for metric in METRICS:
            by_prompt = [
                sum(runs[name]["by_key"][key][metric] - baseline["by_key"][key][metric]
                    for key in keys if key[0] == prompt_id)
                / sum(key[0] == prompt_id for key in keys)
                for prompt_id in prompt_ids
            ]
            difference[metric] = {
                "mean": sum(by_prompt) / len(by_prompt),
                "bootstrap_95_percent_ci": interval(by_prompt, seed=args.seed),
            }
        result["paired_difference_vs_baseline"][name] = difference
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "models"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
