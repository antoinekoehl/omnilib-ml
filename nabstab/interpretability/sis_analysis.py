"""
SIS (Sufficient Input Subsets) analysis for CNN fitness classifiers.

This module provides wrapper functions for applying the SIS algorithm
to our nanobody stability prediction models.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from nabstab.models.fitness_classifier import OmnilibStabilityPredictor
from nabstab.interpretability.shared import (
    CaptumCNNWrapper,
    sequence_to_onehot,
    string_to_tensor,
)
from nabstab.interpretability import sis


def create_sis_wrapper(
    model: OmnilibStabilityPredictor,
    device: torch.device,
    threshold: float
):
    """
    Create wrapper function for SIS that handles shape conventions.

    SIS works with inputs of shape (seq_len, 21).
    Model expects (batch, 21, seq_len).

    For threshold < 0.5 (low stability), returns 1-p so SIS finds
    features driving low predictions.

    Args:
        model: Trained OmnilibStabilityPredictor
        device: Computation device
        threshold: Decision threshold. If < 0.5, predictions are inverted.

    Returns:
        Function f(batch_onehot) -> predictions array
    """
    wrapped = CaptumCNNWrapper(model)
    wrapped.to(device)
    wrapped.eval()

    invert = threshold < 0.5  # For low stability, invert predictions

    def f(batch_onehot):
        # batch_onehot: (batch, seq_len, 21) from SIS
        # Convert to (batch, 21, seq_len) for model
        if isinstance(batch_onehot, np.ndarray):
            batch_onehot = torch.from_numpy(batch_onehot).float()

        x = batch_onehot.transpose(1, 2).to(device)  # (batch, 21, seq_len)

        with torch.no_grad():
            logits = wrapped(x)
            probs = torch.sigmoid(logits).squeeze(-1)

        result = probs.cpu().numpy()
        if invert:
            result = 1.0 - result  # Invert for low stability

        return result

    return f


def verify_masked_baseline(
    model: OmnilibStabilityPredictor,
    seq_len: int = 48,
    device: torch.device = None
) -> float:
    """
    Verify that fully masked input gives ~0.5 prediction.

    This is critical for SIS to work correctly. The fully masked input
    (uniform distribution over amino acids) should yield a prediction
    near random chance (0.5).

    Args:
        model: Trained model
        seq_len: Sequence length
        device: Computation device

    Returns:
        Prediction value for fully masked input
    """
    if device is None:
        device = next(model.parameters()).device

    # Use threshold=0.5 (no inversion) just to check raw prediction
    f = create_sis_wrapper(model, device, threshold=0.5)

    # Create fully masked input: uniform 1/21 at each position
    fully_masked = np.full((1, seq_len, 21), 1.0/21, dtype=np.float32)

    pred = f(fully_masked)[0]
    print(f"Fully masked prediction: {pred:.4f}")

    if not (0.4 < pred < 0.6):
        print(f"WARNING: Masked baseline prediction {pred:.4f} is not near 0.5")
        print("SIS results may not be meaningful.")

    return pred


def run_sis_single(
    model: OmnilibStabilityPredictor,
    sequence_onehot: np.ndarray,
    threshold: float,
    device: torch.device = None
) -> List[sis.SISResult]:
    """
    Run SIS on a single sequence.

    Args:
        model: Trained model
        sequence_onehot: One-hot encoded sequence (seq_len, 21)
        threshold: Decision threshold (tau). Use >= 0.5 for high stability,
                   < 0.5 for low stability features.
        device: Computation device

    Returns:
        List of SISResult objects from the SIS collection
    """
    if device is None:
        device = next(model.parameters()).device

    f = create_sis_wrapper(model, device, threshold)

    seq_len, alphabet_size = sequence_onehot.shape

    # Fully masked: uniform distribution
    fully_masked = np.full((seq_len, alphabet_size), 1.0/alphabet_size, dtype=np.float32)

    # Mask broadcasts over axis 1 (AA dimension) - masks entire positions
    initial_mask = sis.make_empty_boolean_mask_broadcast_over_axis(
        sequence_onehot.shape, axis=1
    )

    # For SIS, tau is always the "positive" threshold
    # When threshold < 0.5, we inverted predictions, so tau = 1 - threshold
    tau = threshold if threshold >= 0.5 else (1.0 - threshold)

    # Run SIS collection
    collection = sis.sis_collection(
        f, tau, sequence_onehot, fully_masked, initial_mask=initial_mask
    )

    return collection


def sis_to_position_counts(
    sequence_onehot: np.ndarray,
    sis_collection: List[sis.SISResult]
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Convert SIS results to position and AA-position counts.

    Args:
        sequence_onehot: One-hot encoded sequence (seq_len, 21)
        sis_collection: List of SISResult objects

    Returns:
        Tuple of:
            - position_counts: Array of shape (seq_len,) with count per position
            - aa_position_counts: Dict mapping (position, aa_idx) -> count
    """
    seq_len = sequence_onehot.shape[0]
    position_counts = np.zeros(seq_len, dtype=int)
    aa_position_counts = {}

    for sis_result in sis_collection:
        for pos_idx in sis_result.sis:
            pos = pos_idx[0]  # Position index (sis indices are arrays)
            position_counts[pos] += 1

            # Get the amino acid at this position
            aa_idx = sequence_onehot[pos].argmax()
            key = (int(pos), int(aa_idx))
            aa_position_counts[key] = aa_position_counts.get(key, 0) + 1

    return position_counts, aa_position_counts


def run_sis_batch(
    model: OmnilibStabilityPredictor,
    sequences: List[str],
    threshold: float,
    device: torch.device = None,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """
    Run SIS on multiple sequences.

    Args:
        model: Trained model
        sequences: List of padded sequence strings
        threshold: Decision threshold (0.9 for high stability, 0.1 for low stability)
        device: Computation device
        verbose: Print progress

    Returns:
        Dict mapping sequence -> (position_counts, aa_position_counts)
    """
    if device is None:
        device = next(model.parameters()).device

    results = {}

    for i, seq in enumerate(sequences):
        if verbose and i % 100 == 0:
            print(f"Processing sequence {i}/{len(sequences)}")

        # Convert to one-hot (seq_len, 21)
        seq_tensor = string_to_tensor(seq)  # (1, seq_len)
        seq_onehot = sequence_to_onehot(seq_tensor)  # (1, 21, seq_len)
        seq_onehot = seq_onehot[0].T.numpy()  # (seq_len, 21)

        collection = run_sis_single(model, seq_onehot, threshold, device)
        pos_counts, aa_pos_counts = sis_to_position_counts(seq_onehot, collection)

        results[seq] = (pos_counts, aa_pos_counts)

    return results
