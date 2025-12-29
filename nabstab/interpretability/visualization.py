"""
Visualization tools for interpretability results.

This module provides plotting functions for DeepLIFT attributions,
position importance, and comparison heatmaps.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from nabstab.constants import AA2INDEX, ALPHABET


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
