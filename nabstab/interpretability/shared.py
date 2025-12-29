"""
Shared utilities for interpretability methods.

This module provides common utilities used by both DeepLIFT and SIS
interpretability approaches, including the Captum model wrapper and
sequence encoding functions.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nabstab.datasets.classifier_dataset import AA2INDEX, pad_cdr2, pad_internal
from nabstab.models.fitness_classifier import OmnilibStabilityPredictor


class CaptumCNNWrapper(nn.Module):
    """
    Wrapper for Captum that takes one-hot encoded inputs directly.

    The original OmnilibStabilityPredictor does one-hot encoding internally,
    but Captum needs gradients to flow through the input tensor. This wrapper
    accepts pre-encoded one-hot tensors.
    """

    def __init__(self, model: OmnilibStabilityPredictor):
        super().__init__()
        self.feature_extractor = model.feature_extractor
        self.classifier = model.classifier

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with one-hot encoded input.

        Args:
            x_onehot: One-hot encoded tensor of shape (batch, alphabet_size, seq_len)
                      where alphabet_size=21 (20 AAs + gap)

        Returns:
            Logits tensor of shape (batch, 1)
        """
        emb = self.feature_extractor(x_onehot)
        logits = self.classifier(emb)
        return logits


def sequence_to_onehot(
    sequence: torch.Tensor,
    alphabet_size: int = 21
) -> torch.Tensor:
    """
    Convert integer sequence tensor to one-hot encoding.

    Args:
        sequence: Integer tensor of shape (batch, seq_len)
        alphabet_size: Size of alphabet (default 21)

    Returns:
        One-hot tensor of shape (batch, alphabet_size, seq_len)
    """
    onehot = F.one_hot(sequence, num_classes=alphabet_size).float()
    return onehot.transpose(1, 2)  # (batch, alphabet_size, seq_len)


def string_to_tensor(
    sequence: str,
    alphabet: Dict[str, int] = None
) -> torch.Tensor:
    """
    Convert sequence string to integer tensor.

    Args:
        sequence: Amino acid sequence string
        alphabet: Mapping from AA to index (default: AA2INDEX)

    Returns:
        Integer tensor of shape (1, seq_len)
    """
    if alphabet is None:
        alphabet = AA2INDEX

    indices = [alphabet.get(aa, alphabet['-']) for aa in sequence]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


def preprocess_sequence(
    cdr1: str,
    cdr2: str,
    cdr3: str,
    cdr3_target_len: int = 28
) -> str:
    """
    Preprocess CDR sequences with standard padding.

    Args:
        cdr1: CDR1 sequence
        cdr2: CDR2 sequence (will be padded to 13 if needed)
        cdr3: CDR3 sequence (will be internally padded to target length)
        cdr3_target_len: Target length for CDR3 padding

    Returns:
        Full padded sequence string
    """
    cdr2_padded = pad_cdr2(cdr2)
    cdr3_padded = pad_internal(cdr3, cdr3_target_len)
    return cdr1 + cdr2_padded + cdr3_padded
