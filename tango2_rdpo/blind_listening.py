"""Prepare anonymous B/I audio pairs without revealing model identity to raters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    generated = [json.loads(line) for line in args.generation_manifest.read_text().splitlines() if line.strip()]
    pairs = defaultdict(dict)
    for item in generated:
        pairs[(item["id"], item["seed"])][item["model"]] = item
    candidates = [key for key, value in pairs.items() if "baseline" in value and "regularized" in value]
    candidates.sort(key=lambda key: hashlib.sha256(f"{args.seed}:{key}".encode()).hexdigest())
    selected = []
    seen_prompts = set()
    for key in candidates:
        if key[0] in seen_prompts:
            continue
        selected.append(key)
        seen_prompts.add(key[0])
        if len(selected) == args.count:
            break
    if len(selected) < args.count:
        raise ValueError(f"only {len(selected)} distinct prompts with complete B/I pairs are available")
    public = args.output_dir / "participant"
    private = args.output_dir / "private"
    audio_dir = public / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    answer_key = {}
    with (public / "ratings_template.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "rater_id", "pair_id", "prompt", "left_audio", "right_audio",
            "alignment_choice", "naturalness_choice",
        ])
        writer.writeheader()
        for index, key in enumerate(selected, 1):
            pair_id = f"pair_{index:03d}"
            left_model, right_model = ("baseline", "regularized")
            if rng.random() < 0.5:
                left_model, right_model = right_model, left_model
            paths = {}
            for side, model_name in (("L", left_model), ("R", right_model)):
                opaque = hashlib.sha256(f"{args.seed}:{pair_id}:{side}".encode()).hexdigest()[:16]
                name = f"{opaque}.wav"
                shutil.copy2(pairs[key][model_name]["path"], audio_dir / name)
                paths[side] = f"audio/{name}"
            writer.writerow({
                "rater_id": "", "pair_id": pair_id, "prompt": pairs[key]["baseline"]["prompt"],
                "left_audio": paths["L"], "right_audio": paths["R"],
                "alignment_choice": "", "naturalness_choice": "",
            })
            answer_key[pair_id] = {
                "prompt_id": key[0], "seed": key[1], "left_model": left_model,
                "right_model": right_model,
            }
    key_path = private / "answer_key.json"
    key_path.write_text(json.dumps(answer_key, ensure_ascii=False, indent=2) + "\n")
    key_path.chmod(0o600)
    print(f"prepared {len(selected)} blind pairs; distribute only {public}")


if __name__ == "__main__":
    main()
