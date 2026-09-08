"""
Custom Weighted SFTTrainer for class-imbalanced Causal Instruction Tuning.

Overrides compute_loss to apply sample-level inverse class frequency weights
strictly to the completion/response tokens (where labels != -100).
"""

from typing import Any

import torch
import torch.nn as nn
from transformers import Trainer
from trl import SFTTrainer


def compute_weighted_completion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_ids: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute causal autoregressive cross-entropy loss weighted by sample class ID.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        labels: Tensor of shape (batch_size, seq_len) with -100 on prompt tokens
        class_ids: Tensor of shape (batch_size,) with integer class IDs (0-99)
        class_weights: Tensor of shape (num_classes,) containing normalized weights

    Returns:
        Scalar torch.Tensor representing the weighted loss.
    """
    # Shift so tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    batch_size = shift_labels.size(0)
    vocab_size = shift_logits.size(-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    ).view(batch_size, -1)

    # Valid completion mask (ignore -100 prompt tokens)
    mask = (shift_labels != -100).float()
    masked_token_loss = token_loss * mask

    # Average loss over response tokens per sample
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    sample_loss = masked_token_loss.sum(dim=1) / token_counts

    # Apply class weights if available
    if class_ids is not None and class_weights is not None:
        device = sample_loss.device
        weights = class_weights.to(device)
        sample_w = weights[class_ids.to(device)]
        total_w = sample_w.sum().clamp(min=1e-8)
        loss = (sample_loss * sample_w).sum() / total_w
    else:
        loss = sample_loss.mean()

    return loss


class WeightedTrainer(Trainer):
    """
    Hugging Face Trainer subclass incorporating class-weighted cross-entropy loss.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute class-weighted autoregressive completion loss."""
        class_ids = inputs.pop("class_id", None)
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss = compute_weighted_completion_loss(
            logits=logits,
            labels=labels,
            class_ids=class_ids,
            class_weights=self.class_weights,
        )

        return (loss, outputs) if return_outputs else loss


class WeightedSFTTrainer(SFTTrainer):
    """
    SFTTrainer subclass incorporating class-weighted cross-entropy loss.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute class-weighted autoregressive completion loss."""
        class_ids = inputs.pop("class_id", None)
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss = compute_weighted_completion_loss(
            logits=logits,
            labels=labels,
            class_ids=class_ids,
            class_weights=self.class_weights,
        )

        return (loss, outputs) if return_outputs else loss
