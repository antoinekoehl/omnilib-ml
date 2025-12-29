"""
Run SIS interpretability on CNN fitness classifier.

Example usage (high stability features, threshold=0.9):
    python scripts/run_sis_interpretability.py \
        --checkpoint checkpoints/CNN/cnn_24_fc_8.pt \
        --test-data data/test.csv \
        --threshold 0.9 \
        --output results/sis_high_stability.json

Example usage (low stability features, threshold=0.1):
    python scripts/run_sis_interpretability.py \
        --checkpoint checkpoints/CNN/cnn_24_fc_8.pt \
        --test-data data/test.csv \
        --threshold 0.1 \
        --output results/sis_low_stability.json
"""
import argparse
import json
from pathlib import Path

import torch
import pandas as pd

from nabstab.utils import load_model
from nabstab.datasets.classifier_dataset import pad_cdr2, pad_internal
from nabstab.interpretability import verify_masked_baseline, run_sis_batch


def main():
    parser = argparse.ArgumentParser(
        description='Run SIS interpretability on CNN fitness classifier'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to CNN checkpoint')
    parser.add_argument('--test-data', required=True,
                        help='Path to CSV with CDR1, CDR2, CDR3 columns')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Decision threshold. Use 0.9 for high stability, 0.1 for low stability')
    parser.add_argument('--output', required=True,
                        help='Output JSON file path')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Limit number of sequences to process')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_type='cnn', device=device)

    # Verify baseline
    print("Verifying masked baseline prediction...")
    verify_masked_baseline(model, seq_len=48, device=device)

    # Load test data
    print(f"Loading data from {args.test_data}")
    df = pd.read_csv(args.test_data)

    # Prepare sequences
    df['cdr2_padded'] = df['CDR2'].apply(pad_cdr2)
    df['cdr3_padded'] = df['CDR3'].apply(lambda x: pad_internal(x, 28))
    df['seq'] = df['CDR1'] + df['cdr2_padded'] + df['cdr3_padded']

    sequences = df['seq'].tolist()
    if args.max_sequences:
        sequences = sequences[:args.max_sequences]

    print(f"Running SIS on {len(sequences)} sequences with threshold={args.threshold}")
    stability_type = "high" if args.threshold >= 0.5 else "low"
    print(f"Finding features associated with {stability_type} stability")

    # Run SIS
    results = run_sis_batch(model, sequences, args.threshold, device)

    # Convert to JSON-serializable format
    output = {
        'metadata': {
            'threshold': args.threshold,
            'stability_type': stability_type,
            'n_sequences': len(sequences),
            'checkpoint': str(args.checkpoint),
        },
        'sequences': {}
    }

    for seq, (pos_counts, aa_pos_counts) in results.items():
        output['sequences'][seq] = {
            'position_counts': pos_counts.tolist(),
            'aa_position_counts': {f"{p},{a}": c for (p, a), c in aa_pos_counts.items()}
        }

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == '__main__':
    main()
