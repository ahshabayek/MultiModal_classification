"""
Custom Loss Functions for Hateful Memes Classification

Implements:
1. Focal Loss - Handles class imbalance by down-weighting easy examples
2. Label Smoothing Cross Entropy - Prevents overconfidence
3. Combined losses for best performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reduces the loss contribution from easy examples and focuses
    training on hard negatives. Particularly useful for imbalanced
    datasets like Hateful Memes (~35% hateful, 65% not hateful).

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Args:
        alpha: Weighting factor for the rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # Probability of correct class

        # Apply focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting for class balance
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.

    Prevents the model from becoming overconfident by smoothing
    the target distribution. Instead of [0, 1], uses [ε/K, 1-ε+ε/K].

    Args:
        smoothing: Label smoothing factor (default: 0.1)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        num_classes = inputs.size(-1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # Compute cross entropy with smoothed targets
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLabelSmoothingLoss(nn.Module):
    """
    Combined Focal Loss with Label Smoothing.

    Best of both worlds: handles class imbalance (focal) and
    prevents overconfidence (label smoothing).

    Args:
        alpha: Focal loss alpha (class weight)
        gamma: Focal loss gamma (focusing parameter)
        smoothing: Label smoothing factor
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)

        # Cross entropy with smoothed targets
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Focal weight
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (
                1 - targets.float()
            )
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropy(nn.Module):
    """
    Cross Entropy with class weights based on frequency.

    Automatically computes weights inversely proportional to
    class frequency in the training set.

    Args:
        class_counts: Number of samples per class [count_class_0, count_class_1]
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        class_counts: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction

        if class_counts is not None:
            # Compute weights inversely proportional to frequency
            total = class_counts.sum()
            weights = total / (len(class_counts) * class_counts)
            self.register_buffer("weight", weights)
        else:
            self.weight = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction=self.reduction,
        )


def get_loss_function(
    loss_type: str = "focal",
    alpha: float = 0.35,  # ~35% hateful in training set
    gamma: float = 2.0,
    smoothing: float = 0.1,
    class_counts: torch.Tensor = None,
) -> nn.Module:
    """
    Factory function to get a loss function.

    Args:
        loss_type: One of "ce", "focal", "label_smoothing", "focal_smoothing", "weighted"
        alpha: Focal loss alpha (set to minority class ratio)
        gamma: Focal loss gamma
        smoothing: Label smoothing factor
        class_counts: For weighted CE, counts per class

    Returns:
        Loss function module
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif loss_type == "focal_smoothing":
        return FocalLabelSmoothingLoss(alpha=alpha, gamma=gamma, smoothing=smoothing)
    elif loss_type == "weighted":
        return WeightedCrossEntropy(class_counts=class_counts)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
