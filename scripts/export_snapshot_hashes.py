"""Keep verified source hashes without local download paths or signed URLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for key, path in (("audio_alpaca", args.dataset_manifest), ("tango_base", args.model_manifest)):
        source = json.loads(path.read_text())
        result[key] = {
            "repo": source["repo"],
            "revision": source["revision"],
            "files": [
                {"name": item["repo_path"], "bytes": item["bytes"], "sha256": item["sha256"]}
                for item in source["files"]
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {sum(len(x['files']) for x in result.values())} verified file hashes")


if __name__ == "__main__":
    main()
