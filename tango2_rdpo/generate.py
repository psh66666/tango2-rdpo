"""Generate matched Tango/Tango 2 audio for frozen evaluation prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
import yaml
from peft import PeftModel

from tango2_rdpo.model import load_base_components, place_on_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", help="Generate only these configured model names")
    parser.add_argument("--limit", type=int, help="Probe with the first N frozen prompts")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for Tango audio generation")
    config = yaml.safe_load(args.config.read_text())
    prompts = [json.loads(line) for line in args.prompt_manifest.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        prompts = prompts[:args.limit]
    selected_models = args.models or list(config["models"])
    if any(name not in config["models"] for name in selected_models):
        raise ValueError("unknown model name in --models")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "prompt_manifest_sha256": hashlib.sha256(args.prompt_manifest.read_bytes()).hexdigest(),
        "models": selected_models,
        "limit": args.limit,
    }
    run_path = args.output_dir / "generation_run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run_metadata:
        raise ValueError("generation output directory belongs to a different config or prompt set")
    run_path.write_text(json.dumps(run_metadata, indent=2) + "\n")
    output_manifest = args.output_dir / "generation_manifest.jsonl"
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16

    with output_manifest.open("w") as stream:
        for model_name in selected_models:
            adapter = config["models"][model_name]
            components = load_base_components(
                upstream_root=Path(config["upstream_root"]),
                snapshot=Path(config["model_snapshot"]),
                text_encoder=Path(config["text_encoder"]),
                scheduler=Path(config["scheduler"]),
                with_reference=False,
            )
            place_on_device(components, device, dtype)
            # The upstream vocoder converts its output to NumPy, which cannot read bf16.
            components.vae.to(dtype=torch.float32)
            if adapter is not None:
                components.model.unet = PeftModel.from_pretrained(
                    components.model.unet, Path(adapter) / "adapter", is_trainable=False
                )
            components.model.eval()
            components.vae.eval()
            for row in prompts:
                for sample_seed in config["seeds"]:
                    path = args.output_dir / model_name / f"{row['id']}_seed{sample_seed}.wav"
                    if path.exists():
                        info = sf.info(path)
                        if info.samplerate == 16000 and info.channels == 1 and info.frames >= 16000:
                            stream.write(json.dumps({
                                "model": model_name, "id": row["id"], "prompt": row["prompt"],
                                "seed": sample_seed, "path": str(path), "steps": config["steps"],
                                "guidance": config["guidance"],
                            }, ensure_ascii=False) + "\n")
                            continue
                    torch.manual_seed(int(sample_seed))
                    torch.cuda.manual_seed_all(int(sample_seed))
                    with torch.no_grad():
                        latents = components.model.inference(
                            [row["prompt"]], components.model.inference_scheduler,
                            num_steps=int(config["steps"]),
                            guidance_scale=float(config["guidance"]),
                            disable_progress=True,
                        )
                        mel = components.vae.decode_first_stage(latents.float())
                        waveform = components.vae.decode_to_waveform(mel)[0]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, waveform, samplerate=16000, subtype="PCM_16")
                    stream.write(json.dumps({
                        "model": model_name, "id": row["id"], "prompt": row["prompt"],
                        "seed": sample_seed, "path": str(path), "steps": config["steps"],
                        "guidance": config["guidance"],
                    }, ensure_ascii=False) + "\n")
                    stream.flush()
            del components
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
