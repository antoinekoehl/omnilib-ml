import re
import os
from typing import Optional, Tuple, List
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from nabstab.utils import load_model
from nabstab.model import OmnilibStabilityPredictor
from nabstab.dataset import AA2INDEX, pad_cdr2, pad_internal

def write_to_file_for_anarci(
        df: pd.DataFrame,
        filename: str,
        base_sequence: str) -> None:
    output_handle = open(filename, 'w')

    for i, row in tqdm(df.iterrows(), total=len(df)):
        output_handle.write(f'>{i}\n{row["sequence"]}\n')

    output_handle.write(f">base\n{base_sequence}\n")

    output_handle.close()

def get_cdr_columns(all_columns: list) -> Tuple[List[str], List[str], List[str]]:

    '''Get the CDR columns from the list of all columns'''
    
    #aho definitions 
    cdr1_aho = np.arange(29,41)
    cdr2_aho = np.arange(54,70)
    cdr3_aho = np.arange(107,139)

    cdr1_columns = []
    cdr2_columns = []
    cdr3_columns = []

    for c in list(filter(lambda x: x[0].isdigit(), all_columns)):
    #check if letter at the end
        if c[-1].isalpha():
            cdr = c[:-1]
        else:
            cdr = c
        if int(cdr) in cdr1_aho:
            cdr1_columns.append(c)
        elif int(cdr) in cdr2_aho:
            cdr2_columns.append(c)
        elif int(cdr) in cdr3_aho:
            cdr3_columns.append(c)

    #sort the columns based on the numbers
    cdr1_columns = sorted(cdr1_columns, key=lambda x: int(x) if not x[-1].isalpha() else int(x[:-1]))
    cdr2_columns = sorted(cdr2_columns, key=lambda x: int(x) if not x[-1].isalpha() else int(x[:-1]))
    cdr3_columns = sorted(cdr3_columns, key=lambda x: int(x) if not x[-1].isalpha() else int(x[:-1]))

    return cdr1_columns, cdr2_columns, cdr3_columns

def get_cdrs_from_anarci_output(
        filename: str, 
        ) -> pd.DataFrame:

    anarci_df = pd.read_csv(filename)
    cdr1_columns, cdr2_columns, cdr3_columns = get_cdr_columns(anarci_df.columns)
    cdr1 = anarci_df[cdr1_columns].apply(lambda x: ''.join(x.dropna()), axis=1)
    cdr2 = anarci_df[cdr2_columns].apply(lambda x: ''.join(x.dropna()), axis=1)
    cdr3 = anarci_df[cdr3_columns].apply(lambda x: ''.join(x.dropna()), axis=1)

    cdr_df = pd.DataFrame({'cdr1': cdr1, 'cdr2': cdr2, 'cdr3': cdr3})

    return cdr_df, anarci_df, [cdr1_columns, cdr2_columns, cdr3_columns]

def process_nanobody_set(
        input_filename: str,
        anarci_filename: str,
        output_filename: str,
) -> pd.DataFrame:
    
    if output_filename.exists():
        return pd.read_csv(output_filename)
    
    def edit_distance(row, reference_sequence):
        return np.sum(row.values != reference_sequence)
    
    
    df = pd.read_csv(input_filename, sep = ',' if str(input_filename).endswith('.csv') else '\t')
    cdr_df, anarci_df, cdr_columns = get_cdrs_from_anarci_output(anarci_filename)

    all_cdr_columns = [c for sublist in cdr_columns for c in sublist]

    noncdr_columns = [c for c in anarci_df.columns if c not in all_cdr_columns]
    framework = anarci_df[noncdr_columns]
    reference_sequence = framework.iloc[-1].values
     #get edit distance between each row and the reference sequence in the framework (this is the part that's really different)
    framework['edit_distance'] = framework.iloc[:-1].apply(lambda x: edit_distance(x, reference_sequence), axis=1)

    #concat original dataframe with cdr dataframe, but not the last row which is the base sequence (Nb80)
    df_new = pd.concat([df['sequence'], cdr_df[:-1]], axis=1)
    df_new = df_new.applymap(lambda x: x.replace('-', '') if isinstance(x, str) else x)
    df_new['edit_distance'] = framework['edit_distance']

    df_new.to_csv(output_filename, index=False) #save

    return df_new

def filter_df_for_compliance(
        df: pd.DataFrame,
) -> pd.DataFrame:
    #CDR lengths based on model training, and remove sequences with a 'C' in CDR2

    df_lengths = df[(df.cdr1.str.len() == 7) & 
                    (df.cdr2.str.len().between(12,13, inclusive='both'))
                      & (df.cdr3.str.len() <= 28)]
    df_lengths = df_lengths[~df_lengths.cdr2.str.contains('C')]

    return df_lengths

def score_sequences(
        df: pd.DataFrame,
        lrmodel: OmnilibStabilityPredictor,
        cnnmodel: OmnilibStabilityPredictor,
        device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    
    #work already done
    if 'CNN' in df.columns and 'LR' in df.columns:
        return df
    
    #check if there's a padded cdr2 column
    if 'cdr2_padded' in df.columns:
        df['padded_sequence'] = df['cdr1'] + df['cdr2_padded'] + df['cdr3_padded']
    else:
        df['cdr2_padded'] = df['cdr2'].apply(pad_cdr2)
        df['cdr3_padded'] = df['cdr3'].apply(lambda x: pad_internal(x, 28))
        df['padded_sequence'] = df['cdr1'] + df['cdr2_padded'] + df['cdr3_padded']

    numseqs = torch.tensor([[AA2INDEX[aa] for aa in seq] for seq in df['padded_sequence']])
    numseqs = numseqs.to(device)

    with torch.no_grad():
        preds_cnn = cnnmodel.predict(numseqs, return_logits=True)
        preds_lr = lrmodel.predict(numseqs, return_logits=True)

    return preds_cnn, preds_lr


if __name__ == "__main__":

    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'

    input_directory = data_dir / 'patent_nb_data'
    output_directory = data_dir / 'patent_nb_data'

    checkpoint_dir = current_dir.parent / 'model_checkpoints'

    lr_checkpoint = checkpoint_dir / 'LR/20231223_LR.pt'
    cnn_checkpoint = checkpoint_dir / 'CNN/cnn_24_fc_8.pt'

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    with open(input_directory / 'base.fasta', 'r') as f:
        label = next(f).strip()
        base_sequence = next(f).strip()

    input_patent_file =input_directory / 'patent_sequence.tsv'
    patent_anarci_file = input_directory / 'patent_H.csv'
    patent_sequences_file = output_directory / 'patent.fasta'

    input_pdb_file = input_directory / 'structure_sequence.tsv'
    pdb_anarci_file = input_directory / 'structure_H.csv'
    pdb_sequences_file = output_directory / 'pdb.fasta'

    patent_df = pd.read_csv(input_patent_file, sep='\t')
    pdb_df = pd.read_csv(input_pdb_file, sep='\t')
    
    to_eval = ['patent', 'pdb']

    print('Writing sequences to file for ANARCI')

    for ds in to_eval:
        write_to_file_for_anarci(
            eval(f'{ds}_df'),
            eval(f'{ds}_sequences_file'),
            base_sequence
        )

    print('Processing ANARCI output')

    #import the ANARCI output and process the data
    patent_df_cdr = process_nanobody_set(
        input_patent_file,
        patent_anarci_file,
        output_directory / 'patent_cdr.csv'
    )

    pdb_df_cdr = process_nanobody_set(
        input_pdb_file,
        pdb_anarci_file,
        output_directory / 'pdb_cdr.csv'
    )

    print('Scoring sequences')

    #filter for compliance with model
    patent_df_filtered = filter_df_for_compliance(patent_df_cdr)
    pdb_df_filtered = filter_df_for_compliance(pdb_df_cdr)

    #load the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lrmodel = load_model(
        checkpoint_path=lr_checkpoint,
        model_type= 'lr', 
        device= device)
    
    cnnmodel = load_model(
        checkpoint_path=cnn_checkpoint,
        model_type='cnn',
        device=device)
    
    #score the sequences

    lr_patent, cnn_patent = score_sequences(
        patent_df_filtered,
        lrmodel,
        cnnmodel,
        device
    )

    lr_pdb, cnn_pdb = score_sequences(
        pdb_df_filtered,
        lrmodel,
        cnnmodel,
        device
    )

    patent_df_filtered['CNN'] = cnn_patent
    patent_df_filtered['LR'] = lr_patent

    pdb_df_filtered['CNN'] = cnn_pdb
    pdb_df_filtered['LR'] = lr_pdb

    print('Saving scored sequences')

    patent_df_filtered.to_csv(output_directory / 'patent_scored.csv', index=False)
    pdb_df_filtered.to_csv(output_directory / 'pdb_scored.csv', index=False)

