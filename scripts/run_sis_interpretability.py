"""
Run SIS interpretability on CNN fitness classifier with motif clustering.

This script runs Sufficient Input Subsets (SIS) analysis on sequences,
extracts the raw SIS sets, and clusters them to identify representative motifs.

Example usage (high stability features, threshold=0.9):
    python scripts/run_sis_interpretability.py \
        --checkpoint checkpoints/CNN/cnn_24_fc_8.pt \
        --test-data data/test.csv \
        --threshold 0.9 \
        --output results/sis_high_stability.json

Example usage with DBSCAN clustering:
    python scripts/run_sis_interpretability.py \
        --checkpoint checkpoints/CNN/cnn_24_fc_8.pt \
        --test-data data/test.csv \
        --threshold 0.9 \
        --cluster-method dbscan \
        --eps 0.3 \
        --output results/sis_high_stability.json
"""
import argparse
import json
from pathlib import Path

import torch
import pandas as pd

from nabstab.utils import load_model
from nabstab.datasets.classifier_dataset import pad_cdr2, pad_internal
from nabstab.interpretability import (
    verify_masked_baseline,
    run_sis_batch_with_sets,
    exact_match_cluster,
    cluster_dbscan,
    extract_all_motifs,
    summarize_clustering,
    # Visualization
    plot_sis_position_frequency,
    plot_top_motifs,
    plot_motif_heatmap,
)


def format_signature_for_json(sig):
    """Convert tuple signature to JSON-serializable string."""
    return str(list(sig))


def format_motif_for_json(motif):
    """Convert motif dict to JSON-serializable format."""
    return {
        'cluster_id': motif['cluster_id'],
        'size': motif['size'],
        'representative': motif['representative'],
        'element_frequencies': {
            str(k): v for k, v in motif['element_frequencies'].items()
        }
    }


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

    # Clustering options
    parser.add_argument('--cluster-method', choices=['exact', 'dbscan', 'both'],
                        default='both',
                        help='Clustering method: exact (exact-match only), '
                             'dbscan (DBSCAN only), or both (default)')
    parser.add_argument('--eps', type=float, default=0.3,
                        help='DBSCAN epsilon (max distance for neighbors). '
                             'Lower = tighter clusters. Default: 0.3')
    parser.add_argument('--min-samples', type=int, default=2,
                        help='DBSCAN minimum samples per cluster. Default: 2')
    parser.add_argument('--top-n-exact', type=int, default=50,
                        help='Number of top exact-match clusters to save. Default: 50')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating visualization plots')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='Directory for figure output. Default: same as --output')

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

    stability_type = "high" if args.threshold >= 0.5 else "low"
    stability_df = df[df.stability == stability_type]

    sequences = stability_df['seq'].tolist()
    if args.max_sequences:
        sequences = sequences[:args.max_sequences]

    print(f"Running SIS on {len(sequences)} sequences with threshold={args.threshold}")
    print(f"Finding features associated with {stability_type} stability")

    # Run SIS and get raw sets
    sis_results = run_sis_batch_with_sets(model, sequences, args.threshold, device)

    # Flatten all SIS sets for clustering
    all_sis_sets = []
    set_to_sequence = []  # Track which sequence each set came from
    for seq, sis_sets in sis_results.items():
        for sis_set in sis_sets:
            if sis_set:  # Skip empty sets
                all_sis_sets.append(sis_set)
                set_to_sequence.append(seq)

    print(f"\nTotal SIS sets extracted: {len(all_sis_sets)}")

    # Initialize output
    output = {
        'metadata': {
            'threshold': args.threshold,
            'stability_type': stability_type,
            'n_sequences': len(sequences),
            'n_total_sis_sets': len(all_sis_sets),
            'checkpoint': str(args.checkpoint),
            'cluster_method': args.cluster_method,
        },
        'raw_sis_sets': {seq: sets for seq, sets in sis_results.items()},
    }

    # Exact-match clustering
    if args.cluster_method in ['exact', 'both']:
        print("\n--- Exact-match clustering ---")
        exact_counts = exact_match_cluster(all_sis_sets)
        n_unique = len(exact_counts)
        print(f"Unique SIS signatures: {n_unique}")

        top_exact = exact_counts.most_common(args.top_n_exact)
        print(f"\nTop {min(10, len(top_exact))} most frequent SIS sets:")
        for sig, count in top_exact[:10]:
            print(f"  {list(sig)}: {count}")

        output['exact_match'] = {
            'n_unique_signatures': n_unique,
            'top_signatures': [
                {'signature': list(sig), 'count': count}
                for sig, count in top_exact
            ]
        }

    # DBSCAN clustering
    if args.cluster_method in ['dbscan', 'both']:
        print(f"\n--- DBSCAN clustering (eps={args.eps}, min_samples={args.min_samples}) ---")

        if len(all_sis_sets) > 0:
            labels, clusters = cluster_dbscan(
                all_sis_sets,
                eps=args.eps,
                min_samples=args.min_samples
            )

            summary = summarize_clustering(labels, clusters)
            print(f"Number of clusters: {summary['n_clusters']}")
            print(f"Clustered: {summary['n_clustered']} ({100*(1-summary['noise_fraction']):.1f}%)")
            print(f"Noise: {summary['n_noise']} ({100*summary['noise_fraction']:.1f}%)")

            if summary['n_clusters'] > 0:
                print(f"Cluster sizes: mean={summary['mean_cluster_size']:.1f}, "
                      f"median={summary['median_cluster_size']:.1f}, "
                      f"max={summary['max_cluster_size']}")

            # Extract motifs
            motifs = extract_all_motifs(all_sis_sets, labels, clusters)

            print(f"\nTop {min(10, len(motifs))} motif clusters:")
            for motif in motifs[:10]:
                print(f"  Cluster {motif['cluster_id']} (size={motif['size']}): "
                      f"{motif['representative']}")

            output['dbscan'] = {
                'parameters': {
                    'eps': args.eps,
                    'min_samples': args.min_samples,
                },
                'summary': summary,
                'motifs': [format_motif_for_json(m) for m in motifs],
            }
        else:
            print("No SIS sets to cluster")
            output['dbscan'] = {
                'parameters': {'eps': args.eps, 'min_samples': args.min_samples},
                'summary': {'n_total': 0, 'n_clusters': 0},
                'motifs': [],
            }

    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {args.output}")

    # Generate visualizations
    if not args.no_plots and len(all_sis_sets) > 0:
        print("\n--- Generating visualizations ---")

        # Determine figures directory
        if args.figures_dir:
            fig_dir = Path(args.figures_dir)
        else:
            fig_dir = Path(args.output).parent

        fig_dir.mkdir(parents=True, exist_ok=True)

        # Base name from output file
        base_name = Path(args.output).stem

        # 1. Position frequency plot
        pos_freq_path = fig_dir / f"{base_name}_position_frequency.pdf"
        plot_sis_position_frequency(
            all_sis_sets,
            title=f"SIS Position Frequency ({stability_type} stability)",
            filename=str(pos_freq_path),
        )
        print(f"Saved position frequency plot to {pos_freq_path}")

        # 2. Motif heatmap (if DBSCAN was run)
        if 'dbscan' in output and output['dbscan'].get('motifs'):
            # Convert motifs back to internal format for plotting
            motifs_for_plot = []
            for m in output['dbscan']['motifs']:
                # Parse element_frequencies back to tuple keys
                elem_freq = {}
                for k, v in m['element_frequencies'].items():
                    # k is like "(5, 'L')" as string
                    pos, aa = eval(k)
                    elem_freq[(pos, aa)] = v
                motifs_for_plot.append({
                    'cluster_id': m['cluster_id'],
                    'size': m['size'],
                    'representative': m['representative'],
                    'element_frequencies': elem_freq,
                })

            # Motif heatmap
            heatmap_path = fig_dir / f"{base_name}_motif_heatmap.pdf"
            plot_motif_heatmap(
                motifs_for_plot,
                title=f"Top Motif Positions ({stability_type} stability)",
                filename=str(heatmap_path),
            )
            print(f"Saved motif heatmap to {heatmap_path}")

            # Top motifs logo plot
            logos_path = fig_dir / f"{base_name}_top_motifs.pdf"
            plot_top_motifs(
                motifs_for_plot,
                n_motifs=20,
                filename=str(logos_path),
            )
            print(f"Saved top motifs logos to {logos_path}")


if __name__ == '__main__':
    main()
