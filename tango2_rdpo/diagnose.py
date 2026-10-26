"""Measure the same frozen preference pairs under fixed noise for each adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from tango2_rdpo.model import load_base_components, place_on_device
from tango2_rdpo.train import audio_latents, forward_loss, read_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[1101, 2202, 3303])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for preference diagnostics")
    config = yaml.safe_load(args.config.read_text())
    rows = read_rows(args.manifest, limit=args.limit, seed=int(config["seed"]))
    components = load_base_components(
        upstream_root=Path(config["upstream_root"]),
        snapshot=Path(config["model_snapshot"]),
        text_encoder=Path(config["text_encoder"]),
        scheduler=Path(config["scheduler"]),
        with_reference=True,
    )
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16
    place_on_device(components, torch.device("cuda:0"), dtype)
    components.model.unet = PeftModel.from_pretrained(
        components.model.unet, args.adapter / "adapter", is_trainable=False
    )
    components.model.eval()
    observations = []
    for row in rows:
        latent_seed = int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:8], "big") % (2**63 - 1)
        latents = audio_latents(
            row, components=components, device=torch.device("cuda:0"),
            stage="dpo", seed=latent_seed,
        )
        for noise_seed in args.noise_seeds:
            with torch.no_grad():
                _, metrics = forward_loss(
                    components, latents, row["prompt"], stage="dpo",
                    beta=float(config["beta"]), lambda_reject=0.0,
                    noise_seed=int(noise_seed) + latent_seed,
                )
            observations.append({"id": row["id"], "noise_seed": noise_seed, **metrics})
    summary = {
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "adapter": str(args.adapter),
        "pairs": len(rows),
        "noise_seeds": args.noise_seeds,
        "metrics_mean": {
            key: sum(row[key] for row in observations) / len(observations)
            for key in ("margin", "preference_correct", "chosen_delta", "rejected_delta", "rejected_proxy_increase_rate")
        },
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "observations"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
