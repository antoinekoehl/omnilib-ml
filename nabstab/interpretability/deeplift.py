"""
DeepLIFT interpretability for CNN fitness classifiers.

This module provides tools for computing DeepLIFT attributions using the
Captum library, with support for shuffled and frequency-based baselines.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from captum.attr import DeepLift

from nabstab.constants import AA2INDEX, ALPHABET
from nabstab.models.fitness_classifier import OmnilibStabilityPredictor
from nabstab.interpretability.shared import (
    CaptumCNNWrapper,
    sequence_to_onehot,
    string_to_tensor,
)


def compute_background_frequencies(
    dataset_path: Union[str, Path],
    exclude_gaps: bool = True
) -> torch.Tensor:
    """
    Compute amino acid frequencies from training data.

    Args:
        dataset_path: Path to CSV file with CDR1, CDR2, CDR3 columns
        exclude_gaps: If True, exclude gap characters from frequency calculation

    Returns:
        Tensor of shape (21,) with frequency per amino acid
    """
    df = pd.read_csv(dataset_path)

    # Concatenate all sequences
    all_seqs = ''.join(df['CDR1'].values) + ''.join(df['CDR2'].values) + ''.join(df['CDR3'].values)

    # Count occurrences
    counts = np.zeros(len(ALPHABET))
    for aa in all_seqs:
        if aa in AA2INDEX:
            counts[AA2INDEX[aa]] += 1

    if exclude_gaps:
        # Set gap count to 0 for normalization, then restore small value
        gap_idx = AA2INDEX['-']
        counts[gap_idx] = 0
        frequencies = counts / counts.sum()
        # Give gaps a small but non-zero frequency
        frequencies[gap_idx] = 1e-6
        frequencies = frequencies / frequencies.sum()
    else:
        frequencies = counts / counts.sum()

    return torch.tensor(frequencies, dtype=torch.float32)


def create_frequency_baseline(
    frequencies: torch.Tensor,
    seq_len: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create baseline tensor where each position has the background distribution.

    Args:
        frequencies: Tensor of shape (21,) with AA frequencies
        seq_len: Sequence length
        device: Target device

    Returns:
        Tensor of shape (1, 21, seq_len) with frequencies replicated at each position
    """
    if device is None:
        device = frequencies.device

    # Expand frequencies to (1, 21, seq_len)
    baseline = frequencies.view(1, -1, 1).expand(1, -1, seq_len).clone()
    return baseline.to(device)


def create_shuffled_baselines(
    sequence_onehot: torch.Tensor,
    n_shuffles: int = 20,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Create n_shuffles random permutations of the input sequence.

    Preserves amino acid composition but destroys positional information.
    Each shuffled sequence is a valid one-hot encoding.

    Args:
        sequence_onehot: One-hot tensor of shape (1, 21, seq_len)
        n_shuffles: Number of shuffled versions to create
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (n_shuffles, 21, seq_len)
    """
    if seed is not None:
        torch.manual_seed(seed)

    batch, alphabet_size, seq_len = sequence_onehot.shape
    assert batch == 1, "Expected single sequence"

    # Convert to indices
    indices = sequence_onehot.argmax(dim=1).squeeze(0)  # (seq_len,)

    shuffled_baselines = []
    for _ in range(n_shuffles):
        # Shuffle positions
        perm = torch.randperm(seq_len)
        shuffled_indices = indices[perm]

        # Convert back to one-hot
        shuffled_onehot = F.one_hot(shuffled_indices, num_classes=alphabet_size)
        shuffled_onehot = shuffled_onehot.float().T  # (21, seq_len)
        shuffled_baselines.append(shuffled_onehot)

    return torch.stack(shuffled_baselines, dim=0)  # (n_shuffles, 21, seq_len)


def compute_deeplift_attributions(
    model: OmnilibStabilityPredictor,
    sequences: torch.Tensor,
    baseline_type: str = 'shuffle',
    n_shuffles: int = 20,
    background_frequencies: Optional[torch.Tensor] = None,
    device: torch.device = None,
    target: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DeepLIFT attributions for CNN model.

    Args:
        model: Trained OmnilibStabilityPredictor with CNN feature extractor
        sequences: Input sequences as integer indices (batch, seq_len)
        baseline_type: 'shuffle' for shuffled sequences, 'frequency' for background dist
        n_shuffles: Number of shuffles if using shuffle baseline
        background_frequencies: Precomputed frequencies if using frequency baseline
        device: Computation device
        target: Target class for attribution (0 for binary classification)

    Returns:
        Tuple of:
            - attributions: Array of shape (batch, 21, seq_len)
            - deltas: Convergence deltas (approximation error)
    """
    if device is None:
        device = next(model.parameters()).device

    # Wrap model for Captum
    wrapped_model = CaptumCNNWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # Convert sequences to one-hot
    sequences = sequences.to(device)
    inputs_onehot = sequence_to_onehot(sequences)  # (batch, 21, seq_len)
    inputs_onehot.requires_grad_(True)

    batch_size, alphabet_size, seq_len = inputs_onehot.shape

    # Initialize DeepLift
    dl = DeepLift(wrapped_model)

    all_attributions = []
    all_deltas = []

    for i in range(batch_size):
        single_input = inputs_onehot[i:i+1]  # (1, 21, seq_len)

        if baseline_type == 'shuffle':
            # Create shuffled baselines
            baselines = create_shuffled_baselines(
                single_input.detach(),
                n_shuffles=n_shuffles
            ).to(device)

            # Compute attributions for each baseline and average
            attr_list = []
            delta_list = []

            for j in range(n_shuffles):
                baseline = baselines[j:j+1]  # (1, 21, seq_len)
                attr, delta = dl.attribute(
                    single_input,
                    baselines=baseline,
                    target=target,
                    return_convergence_delta=True
                )
                attr_list.append(attr.detach().cpu())
                delta_list.append(delta.detach().cpu())

            # Average attributions across shuffles
            avg_attr = torch.stack(attr_list).mean(dim=0)
            avg_delta = torch.stack(delta_list).mean(dim=0)

            all_attributions.append(avg_attr)
            all_deltas.append(avg_delta)

        elif baseline_type == 'frequency':
            if background_frequencies is None:
                raise ValueError("background_frequencies required for frequency baseline")

            baseline = create_frequency_baseline(
                background_frequencies,
                seq_len,
                device=device
            )

            attr, delta = dl.attribute(
                single_input,
                baselines=baseline,
                target=target,
                return_convergence_delta=True
            )

            all_attributions.append(attr.detach().cpu())
            all_deltas.append(delta.detach().cpu())

        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")

    attributions = torch.cat(all_attributions, dim=0).numpy()
    deltas = torch.cat(all_deltas, dim=0).numpy()

    return attributions, deltas


def aggregate_to_position_importance(attributions: np.ndarray) -> np.ndarray:
    """
    Aggregate per-amino-acid attributions to position-level importance.

    For each position, sum absolute attributions across amino acids.

    Args:
        attributions: Array of shape (batch, 21, seq_len) or (21, seq_len)

    Returns:
        Array of shape (batch, seq_len) or (seq_len,)
    """
    return np.abs(attributions).sum(axis=-2)


def aggregate_to_position_signed(
    attributions: np.ndarray,
    sequences: torch.Tensor
) -> np.ndarray:
    """
    Get signed importance at each position based on the actual amino acid present.

    Args:
        attributions: Array of shape (batch, 21, seq_len)
        sequences: Integer tensor of shape (batch, seq_len)

    Returns:
        Array of shape (batch, seq_len)
    """
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.numpy()

    batch_size, _, seq_len = attributions.shape

    result = np.zeros((batch_size, seq_len))
    for b in range(batch_size):
        for pos in range(seq_len):
            aa_idx = sequences[b, pos]
            result[b, pos] = attributions[b, aa_idx, pos]

    return result


def analyze_sequence(
    model: OmnilibStabilityPredictor,
    sequence: str,
    baseline_type: str = 'shuffle',
    n_shuffles: int = 20,
    background_frequencies: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> Dict[str, np.ndarray]:
    """
    High-level function to analyze a single sequence.

    Args:
        model: Trained OmnilibStabilityPredictor
        sequence: Full padded sequence string (CDR1 + CDR2_padded + CDR3_padded)
        baseline_type: 'shuffle' or 'frequency'
        n_shuffles: Number of shuffles for shuffle baseline
        background_frequencies: Precomputed frequencies for frequency baseline
        device: Computation device

    Returns:
        Dict with:
            - 'sequence': Input sequence
            - 'attributions': Full (21, seq_len) attribution matrix
            - 'position_importance': Aggregated (seq_len,) importance
            - 'position_signed': Signed importance at actual AA positions
            - 'prediction': Model prediction probability
            - 'delta': Convergence delta (approximation error)
    """
    if device is None:
        device = next(model.parameters()).device

    # Convert sequence to tensor
    seq_tensor = string_to_tensor(sequence).to(device)

    # Compute attributions
    attributions, deltas = compute_deeplift_attributions(
        model=model,
        sequences=seq_tensor,
        baseline_type=baseline_type,
        n_shuffles=n_shuffles,
        background_frequencies=background_frequencies,
        device=device,
    )

    # Get model prediction
    with torch.no_grad():
        pred = model.predict(seq_tensor, return_logits=False)

    # Aggregate
    position_importance = aggregate_to_position_importance(attributions[0])
    position_signed = aggregate_to_position_signed(attributions, seq_tensor.cpu())[0]

    return {
        'sequence': sequence,
        'attributions': attributions[0],  # (21, seq_len)
        'position_importance': position_importance,  # (seq_len,)
        'position_signed': position_signed,  # (seq_len,)
        'prediction': float(pred[0, 0]),
        'delta': float(deltas[0]),
    }


def batch_analyze(
    model: OmnilibStabilityPredictor,
    sequences: List[str],
    baseline_type: str = 'shuffle',
    n_shuffles: int = 20,
    background_frequencies: Optional[torch.Tensor] = None,
    device: torch.device = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Analyze multiple sequences and return results as DataFrame.

    Args:
        model: Trained OmnilibStabilityPredictor
        sequences: List of full padded sequence strings
        baseline_type: 'shuffle' or 'frequency'
        n_shuffles: Number of shuffles for shuffle baseline
        background_frequencies: Precomputed frequencies for frequency baseline
        device: Computation device
        batch_size: Batch size for processing

    Returns:
        DataFrame with columns:
            - sequence: Input sequence
            - prediction: Model prediction probability
            - delta: Convergence delta
            - position_importance: List of position importance values
            - mean_importance: Mean position importance
    """
    results = []

    for seq in sequences:
        result = analyze_sequence(
            model=model,
            sequence=seq,
            baseline_type=baseline_type,
            n_shuffles=n_shuffles,
            background_frequencies=background_frequencies,
            device=device,
        )

        results.append({
            'sequence': result['sequence'],
            'prediction': result['prediction'],
            'delta': result['delta'],
            'position_importance': result['position_importance'].tolist(),
            'mean_importance': float(result['position_importance'].mean()),
        })

    return pd.DataFrame(results)
