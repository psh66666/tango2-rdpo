"""Publish split identity and hashes without copying audio or machine-local paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "splits.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "split", "strategy", "prompt_sha256", "chosen_sha256", "rejected_sha256"))
        for split in ("train", "val", "test"):
            with (args.manifest_dir / f"{split}.jsonl").open() as rows:
                for line in rows:
                    row = json.loads(line)
                    writer.writerow((
                        row["id"], split, row["strategy"],
                        hashlib.sha256(row["prompt_normalized"].encode()).hexdigest(),
                        row["chosen"]["sha256"], row["rejected"]["sha256"],
                    ))
    audit = json.loads((args.manifest_dir / "audit.json").read_text())
    audit["leakage_audit"] = json.loads((args.manifest_dir / "leakage_audit.json").read_text())
    (args.output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
