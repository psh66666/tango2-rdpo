"""Generate matched Tango/Tango 2 audio for frozen evaluation prompts."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for Tango audio generation")
    config = yaml.safe_load(args.config.read_text())
    prompts = [json.loads(line) for line in args.prompt_manifest.read_text().splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_dir / "generation_manifest.jsonl"
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16

    with output_manifest.open("w") as stream:
        for model_name, adapter in config["models"].items():
            components = load_base_components(
                upstream_root=Path(config["upstream_root"]),
                snapshot=Path(config["model_snapshot"]),
                text_encoder=Path(config["text_encoder"]),
                scheduler=Path(config["scheduler"]),
                with_reference=False,
            )
            place_on_device(components, device, dtype)
            if adapter is not None:
                components.model.unet = PeftModel.from_pretrained(
                    components.model.unet, Path(adapter) / "adapter", is_trainable=False
                )
            components.model.eval()
            components.vae.eval()
            for row in prompts:
                for sample_seed in config["seeds"]:
                    torch.manual_seed(int(sample_seed))
                    torch.cuda.manual_seed_all(int(sample_seed))
                    with torch.no_grad():
                        latents = components.model.inference(
                            [row["prompt"]], components.model.inference_scheduler,
                            num_steps=int(config["steps"]),
                            guidance_scale=float(config["guidance"]),
                            disable_progress=True,
                        )
                        mel = components.vae.decode_first_stage(latents)
                        waveform = components.vae.decode_to_waveform(mel)[0]
                    path = args.output_dir / model_name / f"{row['id']}_seed{sample_seed}.wav"
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
