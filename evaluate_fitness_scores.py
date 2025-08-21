import os
from pathlib import Path
import logging
import time
from typing import List

import pandas as pd
import numpy as np
import seaborn as sns
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from antpack import SingleChainAnnotator

from nabstab.datasets.classifier_dataset import (
    pad_cdr2,
    pad_internal,
    NbStabilityDataset
)

from nabstab.constants import (
    AA2INDEX,
    CDR1_NUMBERS_STR,
    CDR2_NUMBERS_STR,
    CDR3_NUMBERS_STR
)

from nabstab.utils import (
    load_model,
    plot_scores_cnn,
    plot_scores_lr,
    dms_redesign
)


def load_data_for_scoring(
    fasta_file: str,
    cdr1_numbers: str = CDR1_NUMBERS_STR,
    cdr2_numbers: str = CDR2_NUMBERS_STR,
    cdr3_numbers: str = CDR3_NUMBERS_STR,
) -> pd.DataFrame:
    """
    Load sequences from a FASTA file and prepare them for scoring.
    
    Args:
        fasta_file (str): Path to the FASTA file containing sequences.
        cdr1_numbers (str): CDR1 numbering scheme.
        cdr2_numbers (str): CDR2 numbering scheme.
        cdr3_numbers (str): CDR3 numbering scheme.
        alphabet (dict): Mapping of amino acids to indices.
    
    Returns:
        pd.DataFrame: DataFrame with padded sequences ready for scoring.
    """
    
    annotator = SingleChainAnnotator(
        chains = ["H"],
        scheme = "imgt"
        )
    nb_data = []
    seqs = {r.id: str(r.seq) for r in SeqIO.parse(fasta_file, "fasta")}

    for seq_id, seq in seqs.items():
        results = annotator.analyze_seq(seq)
        labels = annotator.assign_cdr_labels(results[0], results[2])
        cdr1 = [aa for aa,idx in zip(seq, results[0]) if idx in cdr1_numbers]
        cdr2 = [aa for aa,idx in zip(seq, results[0]) if idx in cdr2_numbers]
        cdr3 = [aa for aa,idx in zip(seq, results[0]) if idx in cdr3_numbers]

        nb_data.append({
            'seq_id': seq_id,
            'CDR1': ''.join(cdr1),
            'CDR2': ''.join(cdr2),
            'CDR3': ''.join(cdr3),  # Placeholder, as stability is not known for new sequences
            'length': len(seq),
        })

    df = pd.DataFrame(nb_data)

    #filter for conforming sequences
    df = df[
        (df.CDR1.str.len() == 7) &
        (df.CDR2.str.len().between(12,13,inclusive='both')) &
        (df.CDR3.str.len().between(8,28,inclusive='both'))
    ]

    #pad using the same conventions as the training data
    df['cdr2_padded'] = df.CDR2.apply(pad_cdr2)
    df['cdr3_padded'] = df.CDR3.apply(lambda x: pad_internal(x, 28))
    df['padded_sequence'] = df.CDR1 + df.cdr2_padded + df.cdr3_padded

    return df

def evaluate_fitness_scores(
        models: List[nn.Module],
        model_names: List[str],
        sequences: pd.DataFrame,
        device: torch.device,
        alphabet: dict = AA2INDEX,
        batch_size: int = 256,
) -> pd.DataFrame:
    """
    Evaluate fitness scores for a set of sequences using the provided models.
    Args:
        models (List[nn.Module]): The trained models to use for scoring.
        model_names (List[str]): The names of the models (for result labeling).
        sequences (pd.DataFrame): DataFrame containing sequences with 'padded_sequence' column.
        alphabet (dict): Mapping of amino acids to indices.
        batch_size (int): Batch size for processing sequences.
        device (torch.device): Device to run the model on (CPU or GPU).
    
    Returns:
        pd.DataFrame: DataFrame with sequences and their predicted fitness scores.
    """

    numseqs = torch.vstack([
        torch.tensor([alphabet.get(aa, 20) for aa in seq], dtype=torch.long)
        for seq in sequences['padded_sequence']
    ]).to(device)

    for model in models:
        model.eval()
        all_scores = []

        with torch.no_grad():
            for i in range(0, numseqs.shape[0], batch_size):
                batch = numseqs[i:i+batch_size]
                logits = model.predict(batch, return_logits=True)
                all_scores.append(logits)

        all_scores = np.vstack(all_scores)
    
        sequences[f'{model_names[models.index(model)]}_score'] = all_scores
    return sequences


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting fitness score evaluation...")

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate fitness scores for sequences using trained models.")
    parser.add_argument("--fasta_file", type=str, required=True,
                        help="Path to the FASTA file containing sequences.")
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                        help="Paths to the trained model file(s).")
    parser.add_argument("--model_names", type=str, nargs='+', required=True,
                        help="Names of the models for labeling results.")
    parser.add_argument("--output_directory", type=str, required=True,
                        help="Path to save the output DataFrame with fitness scores.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the evaluation on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for processing sequences.")
    args = parser.parse_args()

    # Load sequences from FASTA file
    sequences = load_data_for_scoring(args.fasta_file)

    logging.info(f"Loaded {len(sequences)} sequences from {args.fasta_file}")
    time_start = time.time()

    # Load models
    all_models = []
    for i, model_path in enumerate(args.model_paths):
        model = load_model(
            checkpoint_path=model_path,
            model_type=args.model_names[i], #cnn or lr
            alphabet=AA2INDEX,
            device=torch.device(args.device)
        )

        all_models.append(model)
    logging.info(f"Loaded {len(all_models)} models for evaluation.")

    # Evaluate fitness scores
    sequences_with_scores = evaluate_fitness_scores(
        models=all_models,
        model_names=args.model_names,
        sequences=sequences,
        device=torch.device(args.device),
        alphabet=AA2INDEX,
        batch_size=args.batch_size
    )

    # Save results to output file
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    output_file = os.path.join(args.output_directory, "fitness_scores.csv")
    sequences_with_scores.to_csv(output_file, index=False)

    time_end = time.time()
    logging.info(f"Evaluation completed in {time_end - time_start:.2f} seconds.")

    logging.info(f"Fitness scores saved to {output_file}")
    logging.info("Fitness score evaluation completed.")