"""Load pinned Tango weights while keeping upstream source unmodified."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model


@dataclass
class Components:
    model: torch.nn.Module
    vae: torch.nn.Module
    stft: torch.nn.Module
    reference_unet: torch.nn.Module | None


def _align_librosa_pad_center() -> None:
    import librosa.util

    current = librosa.util.pad_center
    if getattr(current, "_tango_legacy_call", False):
        return
    size = inspect.signature(current).parameters["size"]
    if size.kind is not inspect.Parameter.KEYWORD_ONLY:
        return

    # Pinned Tango passes size positionally, while Librosa 0.9+ requires size=.
    def pad_center(data, size, **kwargs):
        return current(data, size=size, **kwargs)

    pad_center._tango_legacy_call = True
    librosa.util.pad_center = pad_center


def _upstream_module(upstream_root: Path):
    _align_librosa_pad_center()
    root = upstream_root.resolve()
    source = root / "models.py"
    if not source.is_file():
        raise FileNotFoundError(f"official Tango source missing: {source}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("tango_official_models", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import official Tango models")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_components(
    *, upstream_root: Path,
    snapshot: Path,
    text_encoder: Path,
    scheduler: Path,
    with_reference: bool,
) -> Components:
    upstream = _upstream_module(upstream_root)
    from audioldm.audio.stft import TacotronSTFT
    from audioldm.variational_autoencoder import AutoencoderKL

    snapshot = snapshot.resolve()
    main_config = json.loads((snapshot / "main_config.json").read_text())
    main_config["unet_model_config_path"] = str(upstream_root.resolve() / "configs/diffusion_model_config.json")
    main_config["text_encoder_name"] = str(text_encoder.resolve())
    main_config["scheduler_name"] = str(scheduler.resolve())
    model = upstream.AudioDiffusion(**main_config)
    model.load_state_dict(
        torch.load(snapshot / "pytorch_model_main.bin", map_location="cpu", weights_only=True),
        strict=True,
    )
    reference = copy.deepcopy(model.unet) if with_reference else None
    vae = AutoencoderKL(**json.loads((snapshot / "vae_config.json").read_text()))
    vae.load_state_dict(
        torch.load(snapshot / "pytorch_model_vae.bin", map_location="cpu", weights_only=True),
        strict=True,
    )
    stft = TacotronSTFT(**json.loads((snapshot / "stft_config.json").read_text()))
    stft.load_state_dict(
        torch.load(snapshot / "pytorch_model_stft.bin", map_location="cpu", weights_only=True),
        strict=True,
    )
    model.text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    stft.requires_grad_(False).eval()
    if reference is not None:
        reference.requires_grad_(False).eval()
    return Components(model=model, vae=vae, stft=stft, reference_unet=reference)


def attach_lora(
    model: torch.nn.Module,
    *, rank: int,
    alpha: int,
    adapter_dir: Path | None = None,
) -> None:
    model.unet.enable_gradient_checkpointing()
    if adapter_dir is None:
        adapter_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            bias="none",
        )
        model.unet = get_peft_model(model.unet, adapter_config)
    else:
        model.unet = PeftModel.from_pretrained(model.unet, adapter_dir, is_trainable=True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable <= 0:
        raise RuntimeError("LoRA has no trainable parameters")


def place_on_device(components: Components, device: torch.device, dtype: torch.dtype) -> None:
    components.model.to(device=device, dtype=dtype)
    components.vae.to(device=device, dtype=dtype)
    if components.reference_unet is not None:
        components.reference_unet.to(device=device, dtype=dtype)
    # STFT runs on CPU in the preprocessing path, before the VAE input moves to CUDA.
