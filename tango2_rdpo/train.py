"""Single-GPU LoRA SFT and Tango 2 DPO with an optional rejected-audio term."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from tango2_rdpo.losses import preference_loss
from tango2_rdpo.model import attach_lora, load_base_components, place_on_device


def read_rows(path: Path, *, limit: int | None, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    if limit is not None:
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
        rows = rows[:limit]
    return rows


def sample_seed(seed: int, epoch: int, identifier: str) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{identifier}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def audio_latents(row: dict, *, components, device: torch.device, stage: str, seed: int) -> torch.Tensor:
    from tools.torch_tools import wav_to_fbank

    paths = [row["chosen"]["path"]]
    if stage == "dpo":
        paths.append(row["rejected"]["path"])
    mel, _, _ = wav_to_fbank(paths, target_length=1024, fn_STFT=components.stft)
    mel = mel.unsqueeze(1).to(device=device, dtype=next(components.vae.parameters()).dtype)
    with torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        posterior = components.vae.encode_first_stage(mel)
        return components.vae.get_first_stage_encoding(posterior)


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forward_loss(components, latents: torch.Tensor, prompt: str, *, stage: str, beta: float,
                 lambda_reject: float, noise_seed: int):
    model = components.model
    device = latents.device
    hidden, mask = model.encode_text([prompt])
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    timestep = torch.randint(
        0, model.noise_scheduler.config.num_train_timesteps, (1,), device=device, generator=generator
    )
    noise = torch.randn(latents[:1].shape, device=device, dtype=latents.dtype, generator=generator)
    if stage == "dpo":
        hidden = hidden.repeat(2, 1, 1)
        mask = mask.repeat(2, 1)
        timestep = timestep.repeat(2)
        noise = noise.repeat(2, 1, 1, 1)
    noisy = model.noise_scheduler.add_noise(latents, noise, timestep)
    if model.noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif model.noise_scheduler.config.prediction_type == "v_prediction":
        target = model.noise_scheduler.get_velocity(latents, noise, timestep)
    else:
        raise ValueError("unsupported scheduler prediction type")
    prediction = model.unet(
        noisy, timestep, hidden, encoder_attention_mask=mask
    ).sample
    model_mse = per_sample_mse(prediction, target)
    if stage == "sft":
        if model.snr_gamma is not None:
            snr = model.compute_snr(timestep)
            weight = torch.minimum(snr, torch.full_like(snr, model.snr_gamma)) / snr
            model_mse = model_mse * weight
        return model_mse.mean(), {"sft_mse": model_mse.detach().mean().item()}

    if components.reference_unet is None:
        raise RuntimeError("DPO requires a frozen reference UNet")
    with torch.no_grad():
        reference_prediction = components.reference_unet(
            noisy, timestep, hidden, encoder_attention_mask=mask
        ).sample
        reference_mse = per_sample_mse(reference_prediction, target)
    result = preference_loss(
        model_mse[:1], model_mse[1:], reference_mse[:1], reference_mse[1:],
        beta=beta, lambda_reject=lambda_reject,
    )
    metrics = {
        "dpo": result.dpo.detach().item(),
        "rejected_regularizer": result.rejected_regularizer.detach().item(),
        "margin": result.margin.mean().item(),
        "rejected_proxy_increase_rate": result.rejected_proxy_increase_rate.item(),
        "chosen_delta": result.chosen_delta.item(),
        "rejected_delta": result.rejected_delta.item(),
        "preference_correct": float((result.margin > 0).float().mean().item()),
    }
    return result.total, metrics


def save_checkpoint(run_dir: Path, components, optimizer, scheduler, *, epoch: int,
                    next_position: int, updates: int, selected_hash: str,
                    config_hash: str, final: bool) -> None:
    target = run_dir / ("final" if final else "last")
    target.mkdir(parents=True, exist_ok=True)
    components.model.unet.save_pretrained(target / "adapter", safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "next_position": next_position,
            "updates": updates,
            "selected_hash": selected_hash,
            "config_hash": config_hash,
        },
        target / "state.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    stage = config["stage"]
    if stage not in {"sft", "dpo"}:
        raise ValueError("stage must be sft or dpo")
    if stage == "dpo" and args.init is None and not args.resume:
        raise ValueError("DPO must start from the shared SFT adapter")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for the training probe and training")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("GPU does not support bf16; adjust all experiment configs equally")

    rows = read_rows(Path(config["train_manifest"]), limit=config.get("train_limit"), seed=seed)
    selected_hash = hashlib.sha256("\n".join(sorted(row["id"] for row in rows)).encode()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "config": config,
        "selected_ids_sha256": selected_hash,
        "train_manifest_sha256": hashlib.sha256(Path(config["train_manifest"]).read_bytes()).hexdigest(),
        "source_git_sha": source_sha,
        "init": str(args.init) if args.init else None,
        "init_adapter_sha256": sha256_file(args.init / "adapter" / "adapter_model.safetensors") if args.init else None,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "started_at_unix": time.time(),
    }
    run_path = args.output_dir / "run.json"
    if args.resume:
        original = json.loads(run_path.read_text())
        if original["config"] != config or original["selected_ids_sha256"] != selected_hash:
            raise ValueError("resume run metadata mismatch")
    else:
        if run_path.exists():
            raise FileExistsError(f"run metadata already exists: {run_path}")
        run_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    components = load_base_components(
        upstream_root=Path(config["upstream_root"]),
        snapshot=Path(config["model_snapshot"]),
        text_encoder=Path(config["text_encoder"]),
        scheduler=Path(config["scheduler"]),
        with_reference=(stage == "dpo"),
    )
    place_on_device(components, device, dtype)
    adapter_dir = (args.output_dir / "last" / "adapter") if args.resume else (
        args.init / "adapter" if args.init is not None else None
    )
    attach_lora(
        components.model,
        rank=int(config["lora_rank"]),
        alpha=int(config["lora_alpha"]),
        adapter_dir=adapter_dir,
    )
    components.model.text_encoder.eval()
    components.model.unet.train()
    trainable = [parameter for parameter in components.model.unet.parameters() if parameter.requires_grad]
    print(f"trainable LoRA parameters: {sum(p.numel() for p in trainable)}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    accum = int(config["gradient_accumulation"])
    epochs = int(config["epochs"])
    total_updates = math.ceil(len(rows) / accum) * epochs
    warmup = int(config["warmup_updates"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup))
    )
    start_epoch = start_position = updates = 0
    if args.resume:
        state = torch.load(args.output_dir / "last" / "state.pt", map_location="cpu", weights_only=False)
        if state["selected_hash"] != selected_hash:
            raise ValueError("resume data selection mismatch")
        if state["config_hash"] != config_hash:
            raise ValueError("resume config mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, start_position, updates = state["epoch"], state["next_position"], state["updates"]
    optimizer.zero_grad(set_to_none=True)
    accum_count = 0
    started = time.monotonic()
    log_path = args.output_dir / "train.jsonl"
    for epoch in range(start_epoch, epochs):
        order_generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(rows), generator=order_generator).tolist()
        position_start = start_position if epoch == start_epoch else 0
        for position in range(position_start, len(order)):
            row = rows[order[position]]
            row_seed = sample_seed(seed, epoch, row["id"])
            latents = audio_latents(row, components=components, device=device, stage=stage, seed=row_seed)
            loss, metrics = forward_loss(
                components, latents, row["prompt"], stage=stage,
                beta=float(config["beta"]), lambda_reject=float(config["lambda_reject"]),
                noise_seed=row_seed + 1,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at {row['id']}")
            (loss / accum).backward()
            accum_count += 1
            if accum_count < accum and position + 1 != len(order):
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(config["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            accum_count = 0
            record = {
                "epoch": epoch, "position": position, "update": updates, "loss": loss.detach().item(),
                "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.monotonic() - started, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3), **metrics,
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if updates % int(config["checkpoint_every"]) == 0:
                next_epoch, next_position = (epoch + 1, 0) if position + 1 == len(order) else (epoch, position + 1)
                save_checkpoint(args.output_dir, components, optimizer, scheduler,
                                epoch=next_epoch, next_position=next_position,
                                updates=updates, selected_hash=selected_hash,
                                config_hash=config_hash, final=False)
            if args.max_updates is not None and updates >= args.max_updates:
                next_epoch, next_position = (epoch + 1, 0) if position + 1 == len(order) else (epoch, position + 1)
                save_checkpoint(args.output_dir, components, optimizer, scheduler,
                                epoch=next_epoch, next_position=next_position,
                                updates=updates, selected_hash=selected_hash,
                                config_hash=config_hash, final=True)
                print(f"probe stopped at update {updates}/{total_updates}")
                return
    save_checkpoint(args.output_dir, components, optimizer, scheduler, epoch=epochs,
                    next_position=0, updates=updates, selected_hash=selected_hash,
                    config_hash=config_hash, final=True)
    print(f"completed {updates}/{total_updates} optimizer updates")


if __name__ == "__main__":
    main()
