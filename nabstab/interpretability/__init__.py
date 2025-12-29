"""
Interpretability tools for CNN fitness classifiers.

This package provides tools for computing and visualizing feature attributions
for the CNN nanobody stability classifier.

Methods supported:
1. DeepLIFT (using Captum library) with shuffle or frequency baselines
2. Sufficient Input Subsets (SIS) - finds minimal feature subsets that drive predictions

Modules:
    shared: Common utilities (CaptumCNNWrapper, sequence encoding)
    deeplift: DeepLIFT attribution computation and analysis
    sis: Core SIS algorithm (from Google Research)
    sis_analysis: SIS application to our models
    visualization: Plotting functions for interpretability results
"""

# Shared utilities
from nabstab.interpretability.shared import (
    CaptumCNNWrapper,
    sequence_to_onehot,
    string_to_tensor,
    preprocess_sequence,
)

# DeepLIFT
from nabstab.interpretability.deeplift import (
    compute_background_frequencies,
    create_frequency_baseline,
    create_shuffled_baselines,
    compute_deeplift_attributions,
    aggregate_to_position_importance,
    aggregate_to_position_signed,
    analyze_sequence,
    batch_analyze,
)

# SIS analysis
from nabstab.interpretability.sis_analysis import (
    create_sis_wrapper,
    verify_masked_baseline,
    run_sis_single,
    sis_to_position_counts,
    run_sis_batch,
)

# Visualization
from nabstab.interpretability.visualization import (
    plot_deeplift_heatmap,
    plot_position_importance,
    plot_comparison_heatmaps,
)

__all__ = [
    # Shared
    'CaptumCNNWrapper',
    'sequence_to_onehot',
    'string_to_tensor',
    'preprocess_sequence',
    # DeepLIFT
    'compute_background_frequencies',
    'create_frequency_baseline',
    'create_shuffled_baselines',
    'compute_deeplift_attributions',
    'aggregate_to_position_importance',
    'aggregate_to_position_signed',
    'analyze_sequence',
    'batch_analyze',
    # SIS
    'create_sis_wrapper',
    'verify_masked_baseline',
    'run_sis_single',
    'sis_to_position_counts',
    'run_sis_batch',
    # Visualization
    'plot_deeplift_heatmap',
    'plot_position_importance',
    'plot_comparison_heatmaps',
]
