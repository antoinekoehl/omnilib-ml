from functools import partial
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, random_split

AAS = 'ACDEFGHIKLMNPQRSTVWY'
ALPHABET = AAS + '-'
AA2INDEX = {v:i for i,v in enumerate(ALPHABET)}
IDX2AA = {i:v for i,v in enumerate(ALPHABET)}

def pad_cdr2(sequence:str) -> str:
    '''
    Pads CDR2 using known padding position from the library creation stage. 
    '''
    if len(sequence) == 13:
        return sequence
    else:
        return sequence[:6] + '-' + sequence[6:]

def pad_internal(sequence: str, target_len: int) -> str:
    '''
    Applies padding to a CDR sequence to align it to a specific target length.
    The padding is added in the center of the loop in the way the loops were designed.
    '''
    diff = target_len - len(sequence)
    if not diff:
        return sequence
    else:
        return sequence[:len(sequence)//2] + '-' * diff + sequence[len(sequence)//2:]
    
def pad_end(sequence: str, target_len: int) -> str:
    '''
    Applies padding to a CDR sequence to align it to a specific target length.
    The padding is added at the end of the loop in the way the loops were designed.
    '''
    diff = target_len - len(sequence)
    if not diff:
        return sequence
    else:
        return sequence + '-' * diff

class NbStabilityDataset(Dataset):
    def __init__(self, df, alphabet, internal_pad_cdrs = True, cdr3_max_len = 48):
        super().__init__()

        if isinstance(df, str):
            df = pd.read_csv(df)

        cdr3_target = max(df.CDR3.str.len().max(), cdr3_max_len)

        #internally pad the CDRs to max length
        if internal_pad_cdrs:
            df['cdr2_padded'] = df.CDR2.apply(pad_cdr2)
            df['cdr3_padded'] = df.CDR3.apply(lambda x: pad_internal(x, cdr3_target))
            df['padded_sequence'] = df.CDR1 + df.cdr2_padded + df.cdr3_padded
        else:
            full_seq = df.CDR1 + df.CDR2 + df.CDR3
            max_len = full_seq.str.len().max()
            df['padded_sequence'] = full_seq.apply(lambda x: pad_end(x, target_len=max_len))

        self.sequence_length = len(df.padded_sequence.iloc[0])

        self.sequences = df.padded_sequence.values
        self.numseqs = torch.vstack([torch.tensor([alphabet.get(aa, 20) for aa in seq], dtype=torch.long) for seq in self.sequences])
        self.labels = torch.tensor(df.stability == 'high').float()

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.numseqs[idx], self.labels[idx]
    
class CDR1and2Dataset(Dataset):
    def __init__(self, df, alphabet, internal_pad_cdrs = True, cdr3_max_len = 48):
        super().__init__()

        if isinstance(df, str):
            df = pd.read_csv(df)

        cdr3_target = max(df.CDR3.str.len().max(), cdr3_max_len)

        #internally pad the CDRs to max length
        if internal_pad_cdrs:
            df['cdr2_padded'] = df.CDR2.apply(pad_cdr2)
            df['padded_sequence'] = df.CDR1 + df.cdr2_padded
        else:
            full_seq = df.CDR1 + df.CDR2
            max_len = full_seq.str.len().max()
            df['padded_sequence'] = full_seq.apply(lambda x: pad_end(x, target_len=max_len))

        self.sequence_length = len(df.padded_sequence.iloc[0])

        self.sequences = df.padded_sequence.values
        self.numseqs = torch.vstack([torch.tensor([alphabet.get(aa, 20) for aa in seq], dtype=torch.long) for seq in self.sequences])
        self.labels = torch.tensor(df.stability == 'high').float()

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.numseqs[idx], self.labels[idx]
    
class CDRNNDataset(Dataset):
    def __init__(self, datafile):
        if isinstance(datafile, str):
            datafile = pd.read_csv(datafile)

        self.pad_token = 20
        self.eos_token = 21

        #longest sequence is 48 so pad to 49 for the offset for LM
        sequences = [s.ljust(49, '-') for s in datafile.CDR1 + datafile.CDR2 + datafile.CDR3]
        numseqs = torch.tensor([[AA2INDEX[aa] for aa in s] for s in sequences])
        
        numseqs[range(numseqs.shape[0]), (numseqs != self.pad_token).sum(1)] = self.eos_token #nifty way to set the last non-pad token to eos

        self.sequences = numseqs
        self.stability = torch.tensor(datafile.stability == 'high').float() #stability label

    def __len__(self):
        return self.sequences.shape[0]
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.stability[idx]
