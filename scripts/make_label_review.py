"""Select reproducible training pairs for a human Audio-Alpaca label audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.train_manifest.read_text().splitlines() if line.strip()]
    if args.count < 1 or args.count > len(rows):
        raise ValueError("review count must be between 1 and the number of training rows")
    rows.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{row['id']}".encode()).hexdigest())
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "review_template.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "prompt", "strategy", "chosen_audio", "rejected_audio", "label_plausible", "comments"))
        for row in rows[:args.count]:
            names = {}
            for side in ("chosen", "rejected"):
                name = f"{row['id']}_{side}.wav"
                shutil.copy2(row[side]["path"], audio_dir / name)
                names[side] = f"audio/{name}"
            writer.writerow((row["id"], row["prompt"], row["strategy"],
                             names["chosen"], names["rejected"], "", ""))
    print(f"selected {args.count} training pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
