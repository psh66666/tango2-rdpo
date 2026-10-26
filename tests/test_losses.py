import math

import torch

from tango2_rdpo.losses import preference_loss


def test_equal_models_have_log_two_dpo_and_zero_regularizer():
    chosen = torch.tensor([0.1, 0.3])
    rejected = torch.tensor([0.4, 0.6])
    result = preference_loss(chosen, rejected, chosen, rejected, beta=2000, lambda_reject=1)
    assert torch.allclose(result.dpo, torch.tensor(math.log(2)), atol=1e-6)
    assert result.rejected_regularizer.item() == 0


def test_rejected_regularizer_only_activates_for_better_rejected_fit():
    chosen = torch.tensor([0.1])
    reference_rejected = torch.tensor([0.4])
    increased_fit = preference_loss(
        chosen, torch.tensor([0.3]), chosen, reference_rejected, beta=2, lambda_reject=1
    )
    decreased_fit = preference_loss(
        chosen, torch.tensor([0.5]), chosen, reference_rejected, beta=2, lambda_reject=1
    )
    assert increased_fit.rejected_regularizer.item() > 0
    assert decreased_fit.rejected_regularizer.item() == 0


def test_zero_lambda_preserves_original_tango2_formula():
    model_chosen = torch.tensor([0.22], requires_grad=True)
    model_rejected = torch.tensor([0.28], requires_grad=True)
    ref_chosen = torch.tensor([0.2])
    ref_rejected = torch.tensor([0.3])
    result = preference_loss(
        model_chosen, model_rejected, ref_chosen, ref_rejected, beta=2000, lambda_reject=0
    )
    original_margin = -1000 * ((model_chosen - model_rejected) - (ref_chosen - ref_rejected))
    original = -torch.nn.functional.logsigmoid(original_margin).mean()
    assert torch.allclose(result.total, original)
