"""Extract Audio-Alpaca and freeze prompt-grouped train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


SEED = 20261023
DATASET_REVISION = "f41bb361536e53ba064ff5372c8057d4851a8523"


def normalized_prompt(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio(cell: dict) -> tuple[np.ndarray, int]:
    audio_bytes = cell.get("bytes")
    if audio_bytes is None:
        raise ValueError("embedded audio bytes missing")
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("empty or non-finite audio")
    if sample_rate != 16000:
        # The official Tango 2 preprocessing reads every file at 16 kHz.
        import torch
        import torchaudio

        tensor = torch.from_numpy(waveform.copy())
        waveform = torchaudio.functional.resample(tensor, sample_rate, 16000).numpy()
        sample_rate = 16000
    if len(waveform) < 1600:
        raise ValueError("audio shorter than 0.1 second")
    return waveform, sample_rate


def export_audio(cell: dict, path: Path) -> dict:
    waveform, sample_rate = decode_audio(cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".wav.part")
        sf.write(temporary, waveform, sample_rate, format="WAV", subtype="PCM_16")
        temporary.replace(path)
    info = sf.info(path)
    if info.samplerate != 16000 or info.channels != 1 or info.frames < 1600:
        raise ValueError("invalid exported WAV")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "duration_s": round(info.duration, 3),
        "sample_rate": info.samplerate,
    }


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def root(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        a, b = self.root(left), self.root(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def group_records(records: list[dict]) -> tuple[UnionFind, list[dict]]:
    union = UnionFind(len(records))
    seen_prompts: dict[str, int] = {}
    seen_audio: dict[str, int] = {}
    for index, row in enumerate(records):
        prompt = row["prompt_normalized"]
        if prompt in seen_prompts:
            union.union(index, seen_prompts[prompt])
        seen_prompts.setdefault(prompt, index)
        for key in ("chosen", "rejected"):
            audio_hash = row[key]["sha256"]
            if audio_hash in seen_audio:
                union.union(index, seen_audio[audio_hash])
            seen_audio.setdefault(audio_hash, index)

    # Candidate blocking avoids a quadratic comparison over all 15k prompts.
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        tokens = row["prompt_normalized"].split()
        keys = {" ".join(tokens[:3]), " ".join(tokens[-3:])}
        for key in keys:
            if len(key) >= 8:
                buckets[key].append(index)
    reviewed: set[tuple[int, int]] = set()
    near_pairs: list[dict] = []
    for candidates in buckets.values():
        if len(candidates) > 250:
            continue
        for position, left in enumerate(candidates):
            for right in candidates[position + 1 :]:
                pair = (min(left, right), max(left, right))
                if pair in reviewed:
                    continue
                reviewed.add(pair)
                a, b = records[left]["prompt_normalized"], records[right]["prompt_normalized"]
                if a == b:
                    continue
                token_a, token_b = set(a.split()), set(b.split())
                token_overlap = len(token_a & token_b) / max(1, len(token_a | token_b))
                if token_overlap < 0.75:
                    continue
                similarity = SequenceMatcher(None, a, b).ratio()
                if similarity >= 0.86:
                    union.union(left, right)
                    near_pairs.append({"left": records[left]["id"], "right": records[right]["id"], "similarity": round(similarity, 4)})
    return union, near_pairs


def assign_splits(records: list[dict], union: UnionFind) -> dict[str, int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union.root(index)].append(index)
    ordered = sorted(groups.values(), key=lambda indices: hashlib.sha256(
        f"{SEED}:{min(records[i]['id'] for i in indices)}".encode()
    ).hexdigest())
    target = {"train": 0.8 * len(records), "val": 0.1 * len(records), "test": 0.1 * len(records)}
    counts = Counter()
    for indices in ordered:
        split = max(target, key=lambda name: target[name] - counts[name])
        for index in indices:
            records[index]["split"] = split
        counts[split] += len(indices)
    return dict(counts)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    shards = sorted((args.snapshot / "data").glob("train-*.parquet"))
    if len(shards) != 20:
        raise SystemExit(f"expected 20 parquet shards, found {len(shards)}")
    snapshot_manifest = json.loads((args.snapshot / "snapshot_manifest.json").read_text())
    if snapshot_manifest["revision"] != DATASET_REVISION:
        raise SystemExit("dataset revision mismatch")

    records: list[dict] = []
    rejected: list[dict] = []
    for shard in shards:
        row_number = 0
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                identifier = f"{shard.stem}_{row_number:05d}"
                row_number += 1
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or not normalized_prompt(prompt):
                    rejected.append({"id": identifier, "reason": "empty_prompt"})
                    continue
                try:
                    chosen = export_audio(row["chosen"], args.output_dir / "audio" / f"{identifier}_chosen.wav")
                    rejected_audio = export_audio(row["rejected"], args.output_dir / "audio" / f"{identifier}_rejected.wav")
                    if chosen["sha256"] == rejected_audio["sha256"]:
                        raise ValueError("chosen and rejected are identical")
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    rejected.append({"id": identifier, "reason": str(exc)})
                    continue
                records.append({
                    "id": identifier,
                    "prompt": prompt,
                    "prompt_normalized": normalized_prompt(prompt),
                    "strategy": str(row.get("strategy") or "unknown"),
                    "chosen": chosen,
                    "rejected": rejected_audio,
                    "dataset_revision": DATASET_REVISION,
                })
        print(f"{shard.name}: {row_number} rows, accepted total {len(records)}", flush=True)

    union, near_pairs = group_records(records)
    counts = assign_splits(records, union)
    manifests = args.output_dir / "manifests"
    for split in ("train", "val", "test"):
        write_jsonl(manifests / f"{split}.jsonl", [r for r in records if r["split"] == split])
    test_rows = [r for r in records if r["split"] == "test"]
    test_rows.sort(key=lambda row: hashlib.sha256(f"eval:{SEED}:{row['id']}".encode()).hexdigest())
    write_jsonl(
        manifests / "eval_prompts.jsonl",
        [{"id": r["id"], "prompt": r["prompt"], "source": "audio-alpaca-test"} for r in test_rows[:200]],
    )
    write_jsonl(manifests / "rejected.jsonl", rejected)
    write_jsonl(manifests / "near_pairs.jsonl", near_pairs)
    audit = {
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "rows_accepted": len(records),
        "rows_rejected": len(rejected),
        "split_counts": counts,
        "strategies": dict(Counter(r["strategy"] for r in records)),
        "near_prompt_pairs_grouped": len(near_pairs),
        "manifest_sha256": {name: sha256_file(manifests / f"{name}.jsonl") for name in ("train", "val", "test")},
    }
    (manifests / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
