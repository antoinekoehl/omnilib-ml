"""
K-fold cross validation for stability prediction models.

Supports both logistic regression (LR) and CNN models.
Uses clustering-based fold splitting for better generalization evaluation.

Tracks:
- Validation fold AUC per fold
- External test set AUC per fold
- Coefficient values per fold for stability analysis (LR only)
"""

import hashlib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from nabstab.constants import AA2INDEX, CDR1_NUMBERS_STR, CDR2_NUMBERS_STR, CDR3_NUMBERS_STR
from nabstab.datasets.classifier_dataset import NbStabilityDataset
from nabstab.models.fitness_classifier import OmnilibStabilityPredictor, LinearNet, ConvNet, FC
from nabstab.utils import train_model, test_model


def get_hash_seed(phrase: str) -> int:
    """Generate a reproducible random seed from a phrase using MD5 hash."""
    hash_bytes = hashlib.md5(phrase.encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big')


def create_cluster_folds(sequences: np.ndarray, n_folds: int, alphabet: dict, seed: int):
    """
    Create K folds using KMeans clustering on one-hot encoded sequences.

    This ensures sequences in different folds are dissimilar (based on Hamming distance),
    providing a better test of generalization than random splitting.

    Args:
        sequences: Array of padded sequence strings
        n_folds: Number of folds (clusters) to create
        alphabet: Character to index mapping
        seed: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples
    """
    n_samples = len(sequences)
    seq_length = len(sequences[0])
    alphabet_size = len(alphabet)

    # Convert sequences to one-hot encoding
    one_hot = np.zeros((n_samples, seq_length * alphabet_size), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            idx = alphabet.get(aa, alphabet_size - 1)  # Use gap index for unknown
            one_hot[i, j * alphabet_size + idx] = 1.0

    # Run KMeans clustering
    print(f"Clustering {n_samples} sequences into {n_folds} folds using KMeans...")
    kmeans = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(one_hot)

    # Report cluster sizes
    cluster_sizes = [np.sum(cluster_labels == k) for k in range(n_folds)]
    expected_size = n_samples / n_folds
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Expected size per fold: {expected_size:.0f}")

    # Warn about tiny folds (< 10% of expected)
    for k, size in enumerate(cluster_sizes):
        if size < 0.1 * expected_size:
            print(f"WARNING: Fold {k+1} has only {size} samples ({100*size/expected_size:.0f}% of expected)")

    # Create fold splits: each cluster becomes the validation set once
    folds = []
    all_indices = np.arange(n_samples)
    for k in range(n_folds):
        val_mask = cluster_labels == k
        train_indices = all_indices[~val_mask]
        val_indices = all_indices[val_mask]
        folds.append((train_indices, val_indices))

    return folds


def create_fresh_model(model_type: str, sequence_length: int, alphabet: dict, device: torch.device):
    """Create a new model instance with fresh weights.

    Args:
        model_type: Either "lr" for logistic regression or "cnn" for CNN
        sequence_length: Length of padded input sequences
        alphabet: Character to index mapping
        device: Torch device to place model on

    Returns:
        Fresh model instance
    """
    if model_type == "lr":
        fe = nn.Identity()
        cl = LinearNet(sequence_length=sequence_length, alphabet_size=len(alphabet))
    elif model_type == "cnn":
        fe = ConvNet(
            alphabet_size=len(alphabet),
            dim=24,
            ks1=5,
            ks2=9,
        )
        cl = FC(
            alphabet_size=24,
            sequence_length=1,  # pooled
            h_dim=8,
            out_size=1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lr' or 'cnn'.")

    model = OmnilibStabilityPredictor(
        feature_extractor=fe,
        classifier=cl,
        alphabet=alphabet,
    ).to(device)
    return model


def evaluate_on_val_fold(model, val_loader, device):
    """Compute AUC on validation fold."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    return roc_auc_score(all_labels, all_preds)


def extract_coefficients(model) -> np.ndarray:
    """Extract LR coefficients as a 1D array."""
    return model.classifier.linear.weight.detach().cpu().numpy().flatten()


def run_kfold_cv(
    train_data_path: str,
    test_data_path: str,
    model_type: str = "lr",
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed_phrase: str = "cluster k fold",
    device: torch.device = None,
):
    """
    Run K-fold cross validation using clustering-based splits.

    Args:
        train_data_path: Path to training CSV
        test_data_path: Path to test CSV
        model_type: "lr" for logistic regression, "cnn" for CNN
        n_folds: Number of folds
        epochs: Training epochs per fold
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: L2 regularization
        seed_phrase: Phrase for reproducible seed generation
        device: Torch device

    Returns:
        dict with keys:
            - val_aucs: list of validation fold AUCs
            - test_aucs: list of external test set AUCs
            - coefficients: array of shape (n_folds, n_coefficients) [LR only]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = get_hash_seed(seed_phrase)
    print(f"Using seed: {seed} (from phrase: '{seed_phrase}')")
    print(f"Model type: {model_type.upper()}")

    # Load data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # Create full training dataset
    full_dataset = NbStabilityDataset(df=train_df, cdr3_max_len=28)
    sequence_length = full_dataset.sequence_length
    n_samples = len(full_dataset)

    print(f"Training samples: {n_samples}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of folds: {n_folds}")

    # Create cluster-based folds using padded sequences
    folds = create_cluster_folds(
        sequences=full_dataset.sequences,
        n_folds=n_folds,
        alphabet=AA2INDEX,
        seed=seed,
    )

    # Storage for results
    val_aucs = []
    test_aucs = []
    all_coefficients = [] if model_type == "lr" else None

    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")

        # Create subset datasets
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Create fresh model for this fold
        model = create_fresh_model(model_type, sequence_length, AA2INDEX, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
        )

        # Evaluate on validation fold
        val_auc = evaluate_on_val_fold(model, val_loader, device)
        val_aucs.append(val_auc)
        print(f"Validation fold AUC: {val_auc:.4f}")

        # Evaluate on external test set
        test_auc, _, _ = test_model(
            model=model,
            test_data=test_df,
            device=device,
            pad_internally=True,
        )
        test_aucs.append(test_auc)
        print(f"External test AUC: {test_auc:.4f}")

        # Extract coefficients (LR only)
        if model_type == "lr":
            coeffs = extract_coefficients(model)
            all_coefficients.append(coeffs)

    # Build results dict
    results = {
        "val_aucs": val_aucs,
        "test_aucs": test_aucs,
    }

    if model_type == "lr":
        results["coefficients"] = np.stack(all_coefficients)  # shape: (n_folds, n_coefficients)

    return results


def analyze_coefficient_stability(coefficients: np.ndarray, sequence_length: int = 48):
    """
    Analyze stability of coefficients across folds.

    Args:
        coefficients: array of shape (n_folds, n_coefficients)
        sequence_length: length of input sequence (for reshaping)

    Returns:
        dict with coefficient statistics
    """
    n_folds, n_coeffs = coefficients.shape
    alphabet_size = n_coeffs // sequence_length

    # Compute per-position statistics
    coef_mean = coefficients.mean(axis=0)
    coef_std = coefficients.std(axis=0)

    # Coefficient of variation (std/|mean|) where mean != 0
    with np.errstate(divide='ignore', invalid='ignore'):
        coef_cv = np.abs(coef_std / coef_mean)
        coef_cv = np.where(np.isfinite(coef_cv), coef_cv, np.nan)

    return {
        "mean": coef_mean,
        "std": coef_std,
        "cv": coef_cv,
        "n_folds": n_folds,
        "n_coefficients": n_coeffs,
    }


def plot_coefficient_heatmap(
    coefficients: np.ndarray,
    sequence_length: int = 48,
    alphabet: dict = AA2INDEX,
    output_path: str = None,
):
    """
    Plot heatmap of coefficients across folds.

    Each row is a fold, columns are coefficient positions.
    Coefficients are reshaped to (alphabet_size, sequence_length) for each fold.
    """
    n_folds = coefficients.shape[0]
    alphabet_size = len(alphabet)

    # Create figure with subplots for each fold + std dev
    fig, axes = plt.subplots(n_folds + 1, 1, figsize=(16, 3 * (n_folds + 1)))

    # Get amino acid labels
    idx2aa = {v: k for k, v in alphabet.items()}
    aa_labels = [idx2aa[i] for i in range(alphabet_size)]

    # Position labels
    pos_labels = CDR1_NUMBERS_STR + CDR2_NUMBERS_STR + CDR3_NUMBERS_STR

    # Color limits (shared across all heatmaps)
    vmin = coefficients.min()
    vmax = coefficients.max()

    # Plot each fold
    for fold_idx in range(n_folds):
        ax = axes[fold_idx]
        fold_coeffs = coefficients[fold_idx].reshape(alphabet_size, sequence_length)

        sns.heatmap(
            fold_coeffs,
            ax=ax,
            cmap="bwr_r",
            center=0,
            vmin=vmin,
            vmax=vmax,
            xticklabels=pos_labels,
            yticklabels=aa_labels,
            cbar=True,
            cbar_kws={"shrink": 0.5},
        )
        ax.set_title(f"Fold {fold_idx + 1} Coefficients")
        ax.set_xlabel("Position")
        ax.set_ylabel("Amino Acid")

    # Plot std dev across folds
    ax = axes[n_folds]
    coef_std = coefficients.std(axis=0).reshape(alphabet_size, sequence_length)

    sns.heatmap(
        coef_std,
        ax=ax,
        cmap="viridis",
        xticklabels=pos_labels,
        yticklabels=aa_labels,
        cbar=True,
        cbar_kws={"shrink": 0.5},
    )
    ax.set_title("Coefficient Std Dev Across Folds")
    ax.set_xlabel("Position")
    ax.set_ylabel("Amino Acid")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")

    return fig


def print_summary(results: dict, model_type: str):
    """Print summary statistics from K-fold CV."""
    val_aucs = results["val_aucs"]
    test_aucs = results["test_aucs"]

    print("\n" + "=" * 60)
    print(f"K-FOLD CROSS VALIDATION SUMMARY ({model_type.upper()})")
    print("=" * 60)

    print("\nValidation Fold AUCs:")
    for i, auc in enumerate(val_aucs):
        print(f"  Fold {i+1}: {auc:.4f}")
    print(f"  Mean: {np.mean(val_aucs):.4f} +/- {np.std(val_aucs):.4f}")

    print("\nExternal Test Set AUCs:")
    for i, auc in enumerate(test_aucs):
        print(f"  Fold {i+1}: {auc:.4f}")
    print(f"  Mean: {np.mean(test_aucs):.4f} +/- {np.std(test_aucs):.4f}")

    # Coefficient stability (LR only)
    if model_type == "lr" and "coefficients" in results:
        coefficients = results["coefficients"]
        stats = analyze_coefficient_stability(coefficients)
        print("\nCoefficient Stability:")
        print(f"  Number of coefficients: {stats['n_coefficients']}")
        print(f"  Mean std dev: {stats['std'].mean():.6f}")
        print(f"  Max std dev: {stats['std'].max():.6f}")
        print(f"  Min std dev: {stats['std'].min():.6f}")

        # Positions with highest variance
        top_k = 10
        top_var_indices = np.argsort(stats['std'])[-top_k:][::-1]
        print(f"\n  Top {top_k} most variable coefficients (position, aa_idx):")
        for idx in top_var_indices:
            pos = idx % 48
            aa_idx = idx // 48
            print(f"    Position {pos}, AA index {aa_idx}: std = {stats['std'][idx]:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K-fold CV for stability prediction models")
    parser.add_argument("--model-type", type=str, choices=["lr", "cnn"], default="lr",
                        help="Model type: 'lr' for logistic regression, 'cnn' for CNN")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory for plots")
    parser.add_argument("--seed-phrase", type=str, default="cluster k fold", help="Phrase for hash-based seed")
    args = parser.parse_args()

    # Paths
    train_path = "data/model_training_data/ENN_train.csv"
    test_path = "data/model_training_data/ENN_test.csv"

    # Run K-fold CV
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = run_kfold_cv(
        train_data_path=train_path,
        test_data_path=test_path,
        model_type=args.model_type,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed_phrase=args.seed_phrase,
        device=device,
    )

    # Print summary
    print_summary(results, model_type=args.model_type)

    # Plot and save coefficient heatmap (LR only)
    if args.model_type == "lr":
        output_path = f"{args.output_dir}/kfold_{args.model_type}_coefficients.png"
        plot_coefficient_heatmap(
            results["coefficients"],
            sequence_length=48,
            alphabet=AA2INDEX,
            output_path=output_path,
        )

    # Save results to JSON
    to_save = {k: results[k] for k in ["val_aucs", "test_aucs"]}

    with open(f"{args.output_dir}/kfold_{args.model_type}_results.json", "w") as f:
        json.dump(to_save, f, indent=4)

    if args.model_type == "lr":
        plt.show()
