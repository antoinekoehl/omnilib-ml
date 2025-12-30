"""
Run DeepLIFT interpretability on CNN fitness classifier.

Example usage (high stability, shuffle baseline):
    python scripts/run_deeplift_interpretability.py \
        --checkpoint checkpoints/CNN/cnn_24_fc_8.pt \
        --test-data data/test.csv \
        --baseline-type shuffle \
        --n-shuffles 20 \
        --prediction-class high \
        --threshold 0.9 \
        --output results/deeplift_high_shuffle \
        --seed 42

Example usage (low stability, frequency baseline):
    python scripts/run_deeplift_interpretability.py \
        --checkpoint checkpoints/CNN/cnn_24_fc_8.pt \
        --test-data data/test.csv \
        --baseline-type frequency \
        --frequency-data data/train.csv \
        --prediction-class low \
        --threshold 0.1 \
        --output results/deeplift_low_frequency

Example usage (uniform baseline):
    python scripts/run_deeplift_interpretability.py \
        --checkpoint checkpoints/CNN/cnn_24_fc_8.pt \
        --test-data data/test.csv \
        --baseline-type uniform \
        --prediction-class high \
        --threshold 0.9 \
        --output results/deeplift_high_uniform
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nabstab.utils import load_model
from nabstab.datasets.classifier_dataset import pad_cdr2, pad_internal
from nabstab.interpretability import (
    analyze_sequence,
    compute_background_frequencies,
    aggregate_to_position_importance,
)
from nabstab.constants import ALPHABET


def validate_csv_columns(path: Path, required_columns: list[str], name: str) -> pd.DataFrame:
    """Load CSV and validate it has required columns."""
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def prepare_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Pad CDR sequences and create full sequence column."""
    df = df.copy()
    df['cdr2_padded'] = df['CDR2'].apply(pad_cdr2)
    df['cdr3_padded'] = df['CDR3'].apply(lambda x: pad_internal(x, 28))
    df['seq'] = df['CDR1'] + df['cdr2_padded'] + df['cdr3_padded']
    return df


def get_predictions(model, sequences: list[str], device: torch.device) -> np.ndarray:
    """Get model predictions for all sequences."""
    from nabstab.interpretability.shared import string_to_tensor

    predictions = []
    for seq in tqdm(sequences, desc="Getting predictions"):
        seq_tensor = string_to_tensor(seq).to(device)
        with torch.no_grad():
            pred = model.predict(seq_tensor, return_logits=False)
        predictions.append(float(pred[0, 0]))

    return np.array(predictions)


def filter_by_prediction(
    sequences: list[str],
    predictions: np.ndarray,
    prediction_class: str,
    threshold: float,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Filter sequences by prediction class and threshold."""
    if prediction_class == 'high':
        mask = predictions >= threshold
    else:  # low
        mask = predictions <= threshold

    filtered_sequences = [s for s, m in zip(sequences, mask) if m]
    filtered_predictions = predictions[mask]
    indices = np.where(mask)[0]

    return filtered_sequences, filtered_predictions, indices


def main():
    parser = argparse.ArgumentParser(
        description='Run DeepLIFT interpretability on CNN fitness classifier'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to CNN checkpoint')
    parser.add_argument('--test-data', required=True,
                        help='Path to CSV with CDR1, CDR2, CDR3 columns')
    parser.add_argument('--baseline-type', required=True,
                        choices=['shuffle', 'frequency', 'uniform'],
                        help='Baseline type for DeepLIFT')
    parser.add_argument('--prediction-class', required=True,
                        choices=['high', 'low'],
                        help='Filter for high (>=threshold) or low (<=threshold) stability')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Prediction threshold for filtering (e.g., 0.9 for high, 0.1 for low)')
    parser.add_argument('--output', required=True,
                        help='Output path without extension (creates .npz and _metadata.json)')
    parser.add_argument('--n-shuffles', type=int, default=20,
                        help='Number of shuffles for shuffle baseline (default: 20)')
    parser.add_argument('--frequency-data', default=None,
                        help='Path to CSV for computing background frequencies (required for frequency baseline)')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Limit number of sequences to process after filtering')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Validate baseline-type specific requirements
    if args.baseline_type == 'frequency' and args.frequency_data is None:
        raise ValueError("--frequency-data is required when --baseline-type is 'frequency'")

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and validate test data
    print(f"Loading test data from {args.test_data}")
    test_df = validate_csv_columns(
        Path(args.test_data),
        ['CDR1', 'CDR2', 'CDR3'],
        "Test data"
    )
    test_df = prepare_sequences(test_df)
    all_sequences = test_df['seq'].tolist()
    print(f"Loaded {len(all_sequences)} sequences")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_type='cnn', device=device)

    # Get predictions for all sequences
    print("Computing predictions for filtering...")
    all_predictions = get_predictions(model, all_sequences, device)

    # Filter by prediction class
    filtered_sequences, filtered_predictions, filtered_indices = filter_by_prediction(
        all_sequences, all_predictions, args.prediction_class, args.threshold
    )
    print(f"Filtered to {len(filtered_sequences)} sequences with {args.prediction_class} stability "
          f"(threshold {'≥' if args.prediction_class == 'high' else '≤'} {args.threshold})")

    if len(filtered_sequences) == 0:
        raise ValueError(
            f"No sequences passed the filter. "
            f"Prediction range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]"
        )

    # Apply max-sequences limit
    if args.max_sequences and len(filtered_sequences) > args.max_sequences:
        print(f"Limiting to {args.max_sequences} sequences")
        filtered_sequences = filtered_sequences[:args.max_sequences]
        filtered_predictions = filtered_predictions[:args.max_sequences]

    # Prepare baseline
    background_frequencies = None
    if args.baseline_type == 'frequency':
        print(f"Computing background frequencies from {args.frequency_data}")
        freq_df = validate_csv_columns(
            Path(args.frequency_data),
            ['CDR1', 'CDR2', 'CDR3', 'stability'],
            "Frequency data"
        )
        background_frequencies = compute_background_frequencies(
            args.frequency_data,
            exclude_gaps=True
        )
    elif args.baseline_type == 'uniform':
        print("Using uniform background frequencies")
        background_frequencies = torch.ones(len(ALPHABET)) / len(ALPHABET)

    # Run DeepLIFT
    print(f"Running DeepLIFT with {args.baseline_type} baseline on {len(filtered_sequences)} sequences...")

    all_results = []
    for seq in tqdm(filtered_sequences, desc="Computing attributions"):
        result = analyze_sequence(
            model=model,
            sequence=seq,
            baseline_type='shuffle' if args.baseline_type == 'shuffle' else 'frequency',
            n_shuffles=args.n_shuffles,
            background_frequencies=background_frequencies,
            device=device,
        )
        all_results.append(result)

    # Collect results into arrays
    sequences_arr = np.array([r['sequence'] for r in all_results], dtype=object)
    attributions_arr = np.stack([r['attributions'] for r in all_results]).astype(np.float32)
    position_importance_arr = np.stack([r['position_importance'] for r in all_results]).astype(np.float32)
    predictions_arr = np.array([r['prediction'] for r in all_results], dtype=np.float32)
    deltas_arr = np.array([r['delta'] for r in all_results], dtype=np.float32)

    # Report convergence quality
    mean_delta = np.abs(deltas_arr).mean()
    max_delta = np.abs(deltas_arr).max()
    print(f"Convergence deltas - mean: {mean_delta:.6f}, max: {max_delta:.6f}")
    if mean_delta > 0.01:
        print("WARNING: Mean convergence delta > 0.01, attributions may be unreliable")

    # Save outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npz_path = output_path.with_suffix('.npz')
    json_path = output_path.parent / f"{output_path.name}_metadata.json"

    # Save NPZ
    np.savez(
        npz_path,
        sequences=sequences_arr,
        attributions=attributions_arr,
        position_importance=position_importance_arr,
        predictions=predictions_arr,
        deltas=deltas_arr,
    )
    print(f"Saved arrays to {npz_path}")

    # Save metadata JSON
    metadata = {
        'baseline_type': args.baseline_type,
        'prediction_class': args.prediction_class,
        'threshold': args.threshold,
        'n_sequences': len(filtered_sequences),
        'n_shuffles': args.n_shuffles if args.baseline_type == 'shuffle' else None,
        'checkpoint': str(args.checkpoint),
        'test_data': str(args.test_data),
        'frequency_data': str(args.frequency_data) if args.frequency_data else None,
        'seed': args.seed,
        'mean_convergence_delta': float(mean_delta),
        'max_convergence_delta': float(max_delta),
        'npz_file': npz_path.name,
    }

    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {json_path}")


if __name__ == '__main__':
    main()
