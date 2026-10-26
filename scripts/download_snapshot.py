"""Download a pinned public Hugging Face snapshot with SHA-256 verification.

This is useful when the rental instance cannot reach huggingface.co directly.
Run it on a networked machine, then copy the completed directory to the
persistent disk. No model or dataset files are committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


REPOS = {
    "audio-alpaca": (
        "dataset",
        "declare-lab/audio-alpaca",
        "f41bb361536e53ba064ff5372c8057d4851a8523",
        lambda name: name == "README.md" or name.endswith(".parquet"),
    ),
    "tango-base": (
        "model",
        "declare-lab/tango-full-ft-audiocaps",
        "bcdef43a7759667f436cc7444c2f5e7498654c60",
        lambda name: name.endswith((".json", ".bin")) or name == "README.md",
    ),
}


def request(url: str, *, method: str = "GET", start: int | None = None):
    headers = {"User-Agent": "tango2-rdpo-course-project/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers, method=method), timeout=60
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_size(url: str) -> int:
    with request(url, method="HEAD") as response:
        return int(response.headers["Content-Length"])


def download_file(
    url: str, path: Path, expected_size: int, expected_sha: str | None, retries: int = 5
) -> dict:
    size = remote_size(url)
    if size != expected_size:
        raise IOError(f"API and file size disagree for {path.name}: {expected_size} vs {size}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size:
        actual_sha = sha256_file(path)
        if expected_sha is None or actual_sha == expected_sha:
            return {"path": str(path), "bytes": size, "sha256": actual_sha, "status": "existing"}
    part = path.with_name(path.name + ".part")
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset >= size:
            part.unlink()
            offset = 0
        try:
            with request(url, start=offset if offset else None) as response:
                if offset and response.status != 206:
                    part.unlink(missing_ok=True)
                    raise IOError("server ignored Range request")
                with part.open("ab" if offset else "wb") as handle:
                    while block := response.read(8 * 1024 * 1024):
                        handle.write(block)
            if part.stat().st_size != size:
                raise IOError(f"incomplete file: {part.stat().st_size}/{size}")
            actual_sha = sha256_file(part)
            if expected_sha is not None and actual_sha != expected_sha:
                part.unlink()
                raise IOError(f"SHA-256 mismatch for {path.name}")
            part.replace(path)
            return {"path": str(path), "bytes": size, "sha256": actual_sha, "status": "downloaded"}
        except (OSError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise
            print(f"retry {attempt}/{retries}: {path.name}: {exc}", flush=True)
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", choices=REPOS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write-plan", type=Path,
                        help="Write short-lived resolved URLs for a server that cannot reach the HF API")
    parser.add_argument("--plan", type=Path,
                        help="Use a plan written on another machine instead of contacting the HF API")
    args = parser.parse_args()
    if args.write_plan and args.plan:
        parser.error("--write-plan and --plan are mutually exclusive")
    repo_type, repo_id, revision, include = REPOS[args.repo]
    if args.plan:
        plan = json.loads(args.plan.read_text())
        if plan["repo"] != repo_id or plan["revision"] != revision:
            raise RuntimeError("download plan does not match the pinned repository")
        files = plan["files"]
    else:
        api_kind = "datasets" if repo_type == "dataset" else "models"
        resolve_prefix = "datasets/" if repo_type == "dataset" else ""
        url = f"https://huggingface.co/api/{api_kind}/{repo_id}/revision/{revision}?blobs=true"
        with request(url) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision:
            raise RuntimeError("revision mismatch")
        files = []
        for item in sorted(
            (x for x in metadata["siblings"] if include(x["rfilename"])),
            key=lambda entry: entry["rfilename"],
        ):
            name = item["rfilename"]
            file_url = f"https://huggingface.co/{resolve_prefix}{repo_id}/resolve/{revision}/{name}"
            if args.write_plan:
                with request(file_url, method="HEAD") as response:
                    file_url = response.url
            files.append({"name": name, "url": file_url, "size": item["size"],
                          "sha256": item.get("lfs", {}).get("sha256")})
        if args.write_plan:
            args.write_plan.parent.mkdir(parents=True, exist_ok=True)
            args.write_plan.write_text(json.dumps(
                {"repo": repo_id, "revision": revision, "files": files}, indent=2
            ) + "\n")
            args.write_plan.chmod(0o600)
            print(f"wrote {len(files)} expiring URLs to {args.write_plan}; do not commit this file")
            return
    output = args.output_root / args.repo
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(files, 1):
        name = item["name"]
        print(f"[{index}/{len(files)}] {name}", flush=True)
        record = download_file(item["url"], output / name, item["size"], item["sha256"])
        record["repo_path"] = name
        print(f"  {record['status']} {record['bytes']} bytes {record['sha256']}", flush=True)
        records.append(record)
    manifest = {"repo_type": repo_type, "repo": repo_id, "revision": revision, "files": records}
    (output / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
