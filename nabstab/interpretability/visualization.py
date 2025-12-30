"""
Visualization tools for interpretability results.

This module provides plotting functions for DeepLIFT attributions,
position importance, comparison heatmaps, and SIS cluster motifs.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from nabstab.constants import AA2INDEX, ALPHABET


# =============================================================================
# Amino Acid Color Scheme
# =============================================================================

def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color to RGB (0-1 scale)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))


# Amino acid colors by property
AA_COLORS = {
    # Hydrophobic (Nonpolar) - Salomie
    'A': _hex_to_rgb('fdd686'),
    'I': _hex_to_rgb('fdd686'),
    'L': _hex_to_rgb('fdd686'),
    'M': _hex_to_rgb('fdd686'),
    'F': _hex_to_rgb('fdd686'),
    'W': _hex_to_rgb('fdd686'),
    'V': _hex_to_rgb('fdd686'),
    # Positive - Danube
    'R': _hex_to_rgb('6da0cd'),
    'H': _hex_to_rgb('6da0cd'),
    'K': _hex_to_rgb('6da0cd'),
    # Negative - Tapestry
    'D': _hex_to_rgb('b25e7e'),
    'E': _hex_to_rgb('b25e7e'),
    # Polar (Uncharged) - Fern
    'N': _hex_to_rgb('5cb25d'),
    'Q': _hex_to_rgb('5cb25d'),
    'S': _hex_to_rgb('5cb25d'),
    'T': _hex_to_rgb('5cb25d'),
    # Special Cases
    'C': _hex_to_rgb('af93d7'),  # Cold Purple
    'G': _hex_to_rgb('E6ca51'),  # Confetti
    'P': _hex_to_rgb('af93d7'),  # Cold Purple
    'Y': _hex_to_rgb('0f9015'),  # La palma
    # Gap
    '-': _hex_to_rgb('cccccc'),  # Gray
}


def plot_deeplift_heatmap(
    attributions: np.ndarray,
    sequence: str,
    title: str = "DeepLIFT Attributions",
    unseen_mask: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: Tuple[int, int] = (15, 8),
    highlight_sequence: bool = True,
) -> plt.Figure:
    """
    Plot DeepLIFT attribution heatmap similar to existing CNN/LR plots.

    Args:
        attributions: Array of shape (21, seq_len) for single sequence
        sequence: Original sequence string for x-axis labels
        title: Plot title
        unseen_mask: Optional boolean mask for graying out unseen positions
        filename: If provided, save figure to this path
        ax: Optional existing axes to plot on
        vmin, vmax: Color scale limits
        figsize: Figure size if creating new figure
        highlight_sequence: If True, highlight the actual amino acid at each position

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.set_theme(style='white', font_scale=1.0)

    cmap = mpl.colormaps.get_cmap('vlag_r')
    cmap.set_bad(color='gray')

    # Create heatmap
    g = sns.heatmap(
        attributions,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=list(sequence),
        yticklabels=list(ALPHABET),
        annot=True,
        fmt=".2f",
        annot_kws={'rotation': 90, 'fontsize': 6},
        mask=unseen_mask,
        cbar=True,
    )

    # Highlight the actual amino acid at each position
    if highlight_sequence:
        for j, aa in enumerate(sequence):
            if aa in AA2INDEX:
                aa_idx = AA2INDEX[aa]
                ax.add_patch(plt.Rectangle(
                    (j, aa_idx), 1, 1,
                    fill=False, edgecolor='black', lw=2
                ))

    ax.set_title(title)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Amino Acid')

    fig.tight_layout()

    if filename is not None:
        if not isinstance(filename, str):
            filename = str(filename)

        if filename.endswith('.png'):
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(filename, bbox_inches='tight')

    return fig


def plot_position_importance(
    position_importance: np.ndarray,
    sequence: str,
    cdr_boundaries: Optional[Tuple[int, int]] = None,
    title: str = "Position Importance",
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 4),
) -> plt.Figure:
    """
    Bar plot of position-level importance with CDR region annotations.

    Args:
        position_importance: Array of shape (seq_len,)
        sequence: Sequence string for x-axis labels
        cdr_boundaries: Tuple of (CDR1_end, CDR2_end) positions for region shading
        title: Plot title
        filename: If provided, save figure to this path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(len(position_importance))
    colors = ['steelblue' if v >= 0 else 'coral' for v in position_importance]

    ax.bar(x, position_importance, color=colors, edgecolor='black', linewidth=0.5)

    # Add CDR region shading
    if cdr_boundaries is not None:
        cdr1_end, cdr2_end = cdr_boundaries
        ax.axvspan(-0.5, cdr1_end - 0.5, alpha=0.1, color='blue', label='CDR1')
        ax.axvspan(cdr1_end - 0.5, cdr2_end - 0.5, alpha=0.1, color='green', label='CDR2')
        ax.axvspan(cdr2_end - 0.5, len(position_importance) - 0.5, alpha=0.1, color='red', label='CDR3')
        ax.legend(loc='upper right')

    ax.set_xticks(x)
    ax.set_xticklabels(list(sequence), fontsize=8, rotation=90)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Attribution Importance')
    ax.set_title(title)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    fig.tight_layout()

    if filename is not None:
        if not isinstance(filename, str):
            filename = str(filename)

        if filename.endswith('.png'):
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(filename, bbox_inches='tight')

    return fig


def plot_comparison_heatmaps(
    deeplift_attrs: np.ndarray,
    dms_scores: Optional[np.ndarray],
    lr_weights: Optional[np.ndarray],
    sequence: str,
    sequence_tensor: torch.Tensor,
    title_prefix: str = "",
    filename: Optional[str] = None,
    unseen_mask: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of DeepLIFT, DMS, and LR interpretations.

    Args:
        deeplift_attrs: DeepLIFT attributions (21, seq_len)
        dms_scores: DMS scores from dms_redesign (seq_len, 21) or None
        lr_weights: LR weights reshaped to (21, seq_len) or None
        sequence: Sequence string
        sequence_tensor: Integer tensor (1, seq_len) for highlighting
        title_prefix: Prefix for subplot titles
        filename: If provided, save figure
        unseen_mask: Mask for unseen AA positions

    Returns:
        matplotlib Figure
    """
    n_plots = 1 + (dms_scores is not None) + (lr_weights is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 6 * n_plots))

    if n_plots == 1:
        axes = [axes]

    sns.set_theme(style='white', font_scale=1.0)
    cmap = mpl.colormaps.get_cmap('vlag_r')
    cmap.set_bad(color='gray')

    plot_idx = 0

    # DeepLIFT
    ax = axes[plot_idx]
    sns.heatmap(
        deeplift_attrs,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=list(sequence),
        yticklabels=list(ALPHABET),
        annot=True,
        fmt=".2f",
        annot_kws={'rotation': 90, 'fontsize': 6},
        mask=unseen_mask,
        cbar=True,
    )
    for j, aa in enumerate(sequence):
        if aa in AA2INDEX:
            aa_idx = AA2INDEX[aa]
            ax.add_patch(plt.Rectangle((j, aa_idx), 1, 1, fill=False, edgecolor='black', lw=2))
    ax.set_title(f'{title_prefix}DeepLIFT Attributions')
    plot_idx += 1

    # DMS scores
    if dms_scores is not None:
        ax = axes[plot_idx]
        # dms_scores is (seq_len * 21,) or (seq_len, 21), reshape to (21, seq_len)
        if dms_scores.ndim == 1:
            dms_reshaped = dms_scores.reshape(-1, 21).T
        else:
            dms_reshaped = dms_scores.T

        sns.heatmap(
            dms_reshaped,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=list(sequence),
            yticklabels=list(ALPHABET),
            annot=True,
            fmt=".2f",
            annot_kws={'rotation': 90, 'fontsize': 6},
            mask=unseen_mask,
            cbar=True,
        )
        for j, aa in enumerate(sequence):
            if aa in AA2INDEX:
                aa_idx = AA2INDEX[aa]
                ax.add_patch(plt.Rectangle((j, aa_idx), 1, 1, fill=False, edgecolor='black', lw=2))
        ax.set_title(f'{title_prefix}DMS Scores (In-silico Mutagenesis)')
        plot_idx += 1

    # LR weights
    if lr_weights is not None:
        ax = axes[plot_idx]
        sns.heatmap(
            lr_weights,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=list(sequence),
            yticklabels=list(ALPHABET),
            annot=True,
            fmt=".2f",
            annot_kws={'rotation': 90, 'fontsize': 6},
            mask=unseen_mask,
            cbar=True,
        )
        for j, aa in enumerate(sequence):
            if aa in AA2INDEX:
                aa_idx = AA2INDEX[aa]
                ax.add_patch(plt.Rectangle((j, aa_idx), 1, 1, fill=False, edgecolor='black', lw=2))
        ax.set_title(f'{title_prefix}LR Coefficients')

    fig.tight_layout()

    if filename is not None:
        if not isinstance(filename, str):
            filename = str(filename)

        if filename.endswith('.png'):
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(filename, bbox_inches='tight')

    return fig


# =============================================================================
# SIS Cluster Visualization
# =============================================================================

# Page size for publication figures (with margins)
PAGE_WIDTH = 8.0  # inches
PAGE_HEIGHT = 10.0  # inches


def sis_sets_to_frequency_matrix(
    sis_sets: List[List[Tuple[int, str]]],
    seq_len: int = 48,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Convert list of SIS sets to a position x amino acid frequency matrix.

    Args:
        sis_sets: List of SIS sets, each a list of (position, aa) tuples
        seq_len: Length of sequence (default 48)
        normalize: If True, normalize to frequencies; if False, return raw counts

    Returns:
        DataFrame with positions as index and amino acids as columns
    """
    # Count (position, aa) occurrences
    counts = Counter()
    for sis_set in sis_sets:
        for pos, aa in sis_set:
            counts[(pos, aa)] += 1

    # Build matrix
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY-")
    matrix = np.zeros((seq_len, len(amino_acids)))

    for (pos, aa), count in counts.items():
        if aa in amino_acids and pos < seq_len:
            aa_idx = amino_acids.index(aa)
            matrix[pos, aa_idx] = count

    if normalize and len(sis_sets) > 0:
        # Normalize each position by total count at that position
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums

    df = pd.DataFrame(matrix, columns=amino_acids)
    df.index.name = 'position'

    return df


def plot_sis_logo(
    sis_sets: List[List[Tuple[int, str]]],
    seq_len: int = 48,
    title: str = "SIS Motif",
    positions_to_show: Optional[List[int]] = None,
    filename: Optional[str] = None,
    figsize: Tuple[float, float] = (PAGE_WIDTH, 2.0),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a sequence logo for SIS sets using logomaker.

    Args:
        sis_sets: List of SIS sets, each a list of (position, aa) tuples
        seq_len: Total sequence length
        title: Plot title
        positions_to_show: If provided, only show these positions (0-indexed).
                          If None, shows positions that appear in any SIS.
        filename: If provided, save figure to this path
        figsize: Figure size (default fits page width)
        ax: Optional existing axes

    Returns:
        matplotlib Figure
    """
    import logomaker

    # Get frequency matrix
    freq_df = sis_sets_to_frequency_matrix(sis_sets, seq_len, normalize=True)

    # Filter to relevant positions
    if positions_to_show is None:
        # Find positions with any data
        positions_with_data = freq_df.sum(axis=1) > 0
        positions_to_show = list(freq_df[positions_with_data].index)

    if len(positions_to_show) == 0:
        # No data to plot
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.text(0.5, 0.5, "No SIS data", ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig

    # Subset to positions of interest
    logo_df = freq_df.loc[positions_to_show].copy()

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create logo
    logo = logomaker.Logo(
        logo_df,
        ax=ax,
        color_scheme=AA_COLORS,
        font_name='DejaVu Sans Mono',
    )

    # Style
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Position', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)

    # Set x-tick labels to actual positions
    ax.set_xticks(range(len(positions_to_show)))
    ax.set_xticklabels(positions_to_show, fontsize=7)

    fig.tight_layout()

    if filename is not None:
        _save_figure(fig, filename)

    return fig


def plot_sis_position_frequency(
    sis_sets: List[List[Tuple[int, str]]],
    seq_len: int = 48,
    cdr_boundaries: Tuple[int, int] = (7, 20),
    title: str = "SIS Position Frequency",
    filename: Optional[str] = None,
    figsize: Tuple[float, float] = (PAGE_WIDTH, 2.5),
) -> plt.Figure:
    """
    Bar plot showing how often each position appears in SIS sets.

    Args:
        sis_sets: List of SIS sets
        seq_len: Total sequence length
        cdr_boundaries: (CDR1_end, CDR2_end) for region shading
        title: Plot title
        filename: If provided, save figure
        figsize: Figure size (default fits page width)

    Returns:
        matplotlib Figure
    """
    # Count position occurrences
    pos_counts = Counter()
    for sis_set in sis_sets:
        for pos, aa in sis_set:
            pos_counts[pos] += 1

    # Build array
    counts = np.zeros(seq_len)
    for pos, count in pos_counts.items():
        if pos < seq_len:
            counts[pos] = count

    # Normalize to frequency
    if len(sis_sets) > 0:
        freq = counts / len(sis_sets)
    else:
        freq = counts

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(seq_len)
    ax.bar(x, freq, color='steelblue', edgecolor='black', linewidth=0.3)

    # CDR region shading
    cdr1_end, cdr2_end = cdr_boundaries
    ax.axvspan(-0.5, cdr1_end - 0.5, alpha=0.15, color='blue', label='CDR1')
    ax.axvspan(cdr1_end - 0.5, cdr2_end - 0.5, alpha=0.15, color='green', label='CDR2')
    ax.axvspan(cdr2_end - 0.5, seq_len - 0.5, alpha=0.15, color='red', label='CDR3')

    ax.set_xlabel('Position', fontsize=9)
    ax.set_ylabel('Frequency in SIS', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-0.5, seq_len - 0.5)

    fig.tight_layout()

    if filename is not None:
        _save_figure(fig, filename)

    return fig


def plot_top_motifs(
    motifs: List[Dict],
    n_motifs: int = 10,
    seq_len: int = 48,
    filename: Optional[str] = None,
    figsize: Tuple[float, float] = (PAGE_WIDTH, PAGE_HEIGHT),
) -> plt.Figure:
    """
    Plot sequence logos for top N motif clusters (fits on one page).

    Args:
        motifs: List of motif dicts from extract_all_motifs()
        n_motifs: Number of motifs to plot (default 10, fits on page)
        seq_len: Sequence length
        filename: If provided, save figure
        figsize: Figure size (default is full page)

    Returns:
        matplotlib Figure
    """
    import logomaker

    n_to_plot = min(n_motifs, len(motifs))
    if n_to_plot == 0:
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3))
        ax.text(0.5, 0.5, "No motifs to display", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    fig, axes = plt.subplots(n_to_plot, 1, figsize=figsize)
    if n_to_plot == 1:
        axes = [axes]

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY-")

    for i, motif in enumerate(motifs[:n_to_plot]):
        ax = axes[i]

        elem_freq = motif['element_frequencies']

        if not elem_freq:
            ax.text(0.5, 0.5, f"Cluster {motif['cluster_id']}: Empty",
                   ha='center', va='center', fontsize=9)
            ax.axis('off')
            continue

        # Get all positions that appear in this cluster
        positions = sorted(set(pos for pos, aa in elem_freq.keys()))

        if not positions:
            ax.text(0.5, 0.5, f"Cluster {motif['cluster_id']}: No positions",
                   ha='center', va='center', fontsize=9)
            ax.axis('off')
            continue

        # Build frequency matrix for these positions
        matrix = np.zeros((len(positions), len(amino_acids)))
        pos_to_idx = {p: i for i, p in enumerate(positions)}

        total_in_cluster = motif['size']
        for (pos, aa), count in elem_freq.items():
            if pos in pos_to_idx and aa in amino_acids:
                aa_idx = amino_acids.index(aa)
                matrix[pos_to_idx[pos], aa_idx] = count / total_in_cluster

        logo_df = pd.DataFrame(matrix, index=positions, columns=amino_acids)

        # Create logo
        logo = logomaker.Logo(
            logo_df,
            ax=ax,
            color_scheme=AA_COLORS,
            font_name='DejaVu Sans Mono',
        )

        ax.set_title(f"Cluster {motif['cluster_id']} (n={motif['size']})", fontsize=9)
        ax.set_ylabel('Freq', fontsize=8)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, fontsize=7)

        # Only show x-label on bottom plot
        if i == n_to_plot - 1:
            ax.set_xlabel('Position', fontsize=9)

    fig.tight_layout()

    if filename is not None:
        _save_figure(fig, filename)

    return fig


def plot_motif_heatmap(
    motifs: List[Dict],
    seq_len: int = 48,
    n_motifs: int = 10,
    title: str = "Top Motif Positions",
    filename: Optional[str] = None,
    figsize: Tuple[float, float] = (PAGE_WIDTH, 5),
) -> plt.Figure:
    """
    Heatmap showing position coverage across top motifs.

    Rows are motifs, columns are positions. Cell intensity shows
    frequency of that position in the motif.

    Args:
        motifs: List of motif dicts
        seq_len: Sequence length
        n_motifs: Number of motifs to show (default 10)
        title: Plot title
        filename: If provided, save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_to_plot = min(n_motifs, len(motifs))
    if n_to_plot == 0:
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3))
        ax.text(0.5, 0.5, "No motifs to display", ha='center', va='center')
        ax.axis('off')
        return fig

    # Build matrix: motifs x positions
    matrix = np.zeros((n_to_plot, seq_len))

    for i, motif in enumerate(motifs[:n_to_plot]):
        elem_freq = motif['element_frequencies']
        total = motif['size']
        for (pos, aa), count in elem_freq.items():
            if pos < seq_len:
                # Use max frequency if multiple AAs at same position
                matrix[i, pos] = max(matrix[i, pos], count / total)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        ax=ax,
        cmap='Blues',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Position frequency', 'shrink': 0.8},
        yticklabels=[f"C{m['cluster_id']} (n={m['size']})" for m in motifs[:n_to_plot]],
    )

    ax.set_xlabel('Position', fontsize=9)
    ax.set_ylabel('Motif Cluster', fontsize=9)
    ax.set_title(title, fontsize=10)

    fig.tight_layout()

    if filename is not None:
        _save_figure(fig, filename)

    return fig


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Helper to save figure with appropriate format."""
    if not isinstance(filename, str):
        filename = str(filename)

    if filename.endswith('.png'):
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        fig.savefig(filename, bbox_inches='tight')
