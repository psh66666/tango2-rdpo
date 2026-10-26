"""Tango 2 DPO and the proposed rejected-audio proxy regularizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PreferenceLoss:
    total: torch.Tensor
    dpo: torch.Tensor
    rejected_regularizer: torch.Tensor
    margin: torch.Tensor
    rejected_proxy_increase_rate: torch.Tensor
    chosen_delta: torch.Tensor
    rejected_delta: torch.Tensor


def preference_loss(
    model_chosen: torch.Tensor,
    model_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float,
    lambda_reject: float,
) -> PreferenceLoss:
    if beta <= 0 or lambda_reject < 0:
        raise ValueError("beta must be positive and lambda_reject non-negative")
    shapes = {tuple(x.shape) for x in (
        model_chosen, model_rejected, reference_chosen, reference_rejected
    )}
    if len(shapes) != 1:
        raise ValueError("the four per-sample loss arrays must have the same shape")
    if model_chosen.ndim != 1:
        raise ValueError("expected one MSE value per preference pair")

    # Match the Tango 2 implementation exactly when lambda_reject is zero.
    model_difference = model_chosen - model_rejected
    reference_difference = reference_chosen - reference_rejected
    margin = -0.5 * beta * (model_difference - reference_difference)
    dpo = F.softplus(-margin).mean()

    # A lower rejected MSE than the frozen reference is only a likelihood proxy.
    rejected_proxy_gain = 0.5 * beta * (reference_rejected - model_rejected)
    rejected_regularizer = F.relu(rejected_proxy_gain).square().mean()
    total = dpo + lambda_reject * rejected_regularizer
    return PreferenceLoss(
        total=total,
        dpo=dpo,
        rejected_regularizer=rejected_regularizer,
        margin=margin.detach(),
        rejected_proxy_increase_rate=(rejected_proxy_gain.detach() > 0).float().mean(),
        chosen_delta=(model_chosen.detach() - reference_chosen.detach()).mean(),
        rejected_delta=(model_rejected.detach() - reference_rejected.detach()).mean(),
    )
