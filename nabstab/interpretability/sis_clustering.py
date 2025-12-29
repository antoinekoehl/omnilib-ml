"""
SIS set clustering for motif discovery.

This module provides functions for clustering SIS (Sufficient Input Subsets)
to identify representative motifs from interpretability analysis.

Two clustering approaches are supported:
1. Exact-match grouping: Count identical SIS signatures
2. DBSCAN clustering: Group similar SIS sets using Jaccard distance
"""

from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


# Type aliases for clarity
SISSet = List[Tuple[int, str]]  # List of (position, amino_acid) tuples
SISSignature = Tuple[Tuple[int, str], ...]  # Hashable version


def sis_set_to_signature(sis_set: SISSet) -> SISSignature:
    """
    Convert SIS set to hashable signature for exact-match grouping.

    Args:
        sis_set: List of (position, amino_acid) tuples

    Returns:
        Tuple of tuples, sorted by position
    """
    return tuple(sorted(sis_set, key=lambda x: x[0]))


def exact_match_cluster(
    all_sis_sets: List[SISSet]
) -> Counter:
    """
    Group identical SIS sets and count occurrences.

    This is the simplest clustering approach - just count how many times
    each exact SIS signature appears.

    Args:
        all_sis_sets: Flat list of all SIS sets from all sequences

    Returns:
        Counter mapping SIS signature -> count
    """
    counts = Counter()
    for sis_set in all_sis_sets:
        signature = sis_set_to_signature(sis_set)
        counts[signature] += 1
    return counts


def jaccard_distance(
    set_a: Set[Tuple[int, str]],
    set_b: Set[Tuple[int, str]]
) -> float:
    """
    Compute Jaccard distance between two SIS sets.

    Jaccard distance = 1 - |A ∩ B| / |A ∪ B|

    Args:
        set_a: First SIS set as a set of (position, aa) tuples
        set_b: Second SIS set as a set of (position, aa) tuples

    Returns:
        Jaccard distance in [0, 1]. 0 = identical, 1 = no overlap.
    """
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 1.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - (intersection / union)


def compute_distance_matrix(sis_sets: List[SISSet]) -> np.ndarray:
    """
    Compute pairwise Jaccard distance matrix for SIS sets.

    Args:
        sis_sets: List of SIS sets

    Returns:
        Symmetric distance matrix of shape (n, n)
    """
    n = len(sis_sets)
    sets_as_sets = [set(s) for s in sis_sets]

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard_distance(sets_as_sets[i], sets_as_sets[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def cluster_dbscan(
    sis_sets: List[SISSet],
    eps: float = 0.3,
    min_samples: int = 2
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Cluster SIS sets using DBSCAN with precomputed Jaccard distances.

    Args:
        sis_sets: List of SIS sets to cluster
        eps: Maximum distance between two samples to be considered neighbors.
             Lower = tighter clusters. Default 0.3 means sets must share
             ~70% of their elements to be neighbors.
        min_samples: Minimum samples in a neighborhood to form a cluster.

    Returns:
        Tuple of:
            - labels: Cluster label for each SIS set (-1 = noise/outlier)
            - clusters: Dict mapping cluster_id -> list of SIS set indices
    """
    if len(sis_sets) == 0:
        return np.array([]), {}

    dist_matrix = compute_distance_matrix(sis_sets)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(dist_matrix)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return labels, dict(clusters)


def get_cluster_representative(
    cluster_indices: List[int],
    sis_sets: List[SISSet],
    frequency_threshold: float = 0.5
) -> Tuple[SISSet, Dict[Tuple[int, str], int]]:
    """
    Extract representative motif from a cluster of SIS sets.

    The representative consists of (position, aa) pairs that appear in
    more than the specified fraction of cluster members.

    Args:
        cluster_indices: Indices of SIS sets in this cluster
        sis_sets: Full list of all SIS sets
        frequency_threshold: Fraction of cluster members an element must
                            appear in to be included in representative.
                            Default 0.5 = majority voting.

    Returns:
        Tuple of:
            - representative: List of (position, aa) tuples forming the consensus motif
            - element_frequencies: Dict mapping each (position, aa) -> count in cluster
    """
    element_frequencies: Dict[Tuple[int, str], int] = Counter()

    for idx in cluster_indices:
        for pos_aa in sis_sets[idx]:
            element_frequencies[pos_aa] += 1

    # Representative = elements appearing in > threshold fraction of members
    min_count = len(cluster_indices) * frequency_threshold
    representative = [
        pos_aa for pos_aa, count in element_frequencies.items()
        if count > min_count
    ]
    representative = sorted(representative, key=lambda x: x[0])

    return representative, dict(element_frequencies)


def extract_all_motifs(
    sis_sets: List[SISSet],
    labels: np.ndarray,
    clusters: Dict[int, List[int]],
    frequency_threshold: float = 0.5
) -> List[Dict]:
    """
    Extract motif information from all clusters.

    Args:
        sis_sets: Full list of all SIS sets
        labels: Cluster label for each SIS set
        clusters: Dict mapping cluster_id -> list of indices
        frequency_threshold: Threshold for representative extraction

    Returns:
        List of motif dicts, sorted by cluster size (descending).
        Each dict contains:
            - cluster_id: int
            - size: number of SIS sets in cluster
            - representative: consensus (position, aa) pairs
            - element_frequencies: all (position, aa) counts
    """
    motifs = []

    for cluster_id, indices in clusters.items():
        if cluster_id == -1:
            # Skip noise cluster
            continue

        representative, freqs = get_cluster_representative(
            indices, sis_sets, frequency_threshold
        )

        motifs.append({
            'cluster_id': cluster_id,
            'size': len(indices),
            'representative': representative,
            'element_frequencies': freqs
        })

    # Sort by cluster size, largest first
    motifs.sort(key=lambda x: x['size'], reverse=True)

    return motifs


def summarize_clustering(
    labels: np.ndarray,
    clusters: Dict[int, List[int]]
) -> Dict:
    """
    Generate summary statistics for clustering results.

    Args:
        labels: Cluster labels for each SIS set
        clusters: Dict mapping cluster_id -> list of indices

    Returns:
        Dict with summary statistics
    """
    n_clusters = len([c for c in clusters.keys() if c != -1])
    n_noise = len(clusters.get(-1, []))
    n_total = len(labels)
    n_clustered = n_total - n_noise

    cluster_sizes = [len(indices) for cid, indices in clusters.items() if cid != -1]

    return {
        'n_total': n_total,
        'n_clusters': n_clusters,
        'n_clustered': n_clustered,
        'n_noise': n_noise,
        'noise_fraction': n_noise / n_total if n_total > 0 else 0,
        'cluster_sizes': cluster_sizes,
        'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
        'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0,
        'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
    }
