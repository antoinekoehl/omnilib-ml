from typing import Union, Tuple, List, Dict
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from nabstab.datasets.classifier_dataset import (
    pad_cdr2,
    pad_internal,
    AA2INDEX
)
from nabstab.models.fitness_classifier import (
    OmnilibStabilityPredictor,
    ConvNet,
    LinearNet,
    FC,
)

def train_epoch(model, optimizer, loader, device):
    """Train the model for one epoch on the provided data loader."""
    cum_loss = 0
    cum_acc = 0
    model.train()
    for batch in loader:
        seqs, labels = batch
        seqs = seqs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(seqs)
        loss = model.loss_fn(logits, labels.unsqueeze(-1))
        loss.backward()
        optimizer.step()

        class_acc = model.classification_acc(logits, labels)

        cum_loss += loss.item()
        cum_acc += class_acc.item()
    
    return cum_loss / len(loader), cum_acc / len(loader)

def val_epoch(model, loader, device):
    """Validate the model on the provided data loader."""
    cum_loss = 0
    cum_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            seqs, labels = batch
            seqs = seqs.to(device)
            labels = labels.to(device)

            logits = model(seqs)
            loss = model.loss_fn(logits, labels.unsqueeze(-1))
            acc = model.classification_acc(logits, labels)

            cum_loss += loss.item()
            cum_acc += acc.item()
    
    return cum_loss / len(loader), cum_acc / len(loader)

def train_model(model, train_loader, val_loader, device, optimizer, epochs):
    """Train the model for a specified number of epochs."""
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device)
        val_loss, val_acc = val_epoch(model, val_loader, device)
        pbar.set_postfix_str(f"Epoch {epoch}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")
    return val_loss, val_acc

def load_model(
        checkpoint_path: str,
        model_type: str,
        device: torch.device,
        alphabet = AA2INDEX,
        sequence_length = 48) -> torch.nn.Module:
    
    '''Load a model from a checkpoint file'''

    sd = torch.load(checkpoint_path, map_location='cpu')

    if model_type == 'lr':
        fe = nn.Identity() #no feature extraction for linear model
        cl = LinearNet(sequence_length=sequence_length, alphabet_size=len(alphabet))
        model = OmnilibStabilityPredictor(
            feature_extractor=fe,
            classifier=cl,
            alphabet=alphabet
        )
        model.load_state_dict(sd['model_state_dict'])

    elif model_type == 'cnn':
        cnn_dim = sd['params']['dim']
        cnn_ks1 = sd['params']['ks1']
        cnn_ks2 = sd['params']['ks2']

        fe = ConvNet(
            alphabet_size=len(alphabet),
            dim=cnn_dim,
            ks1=cnn_ks1,
            ks2=cnn_ks2
        )

        if 'h_dim' in sd['params']:
            h_dim = sd['params']['h_dim']
            cl = FC(
                alphabet_size=cnn_dim,
                sequence_length=1, #pooled
                h_dim=h_dim,
                out_size=1
            )
        else:
            cl = LinearNet(sequence_length=1, alphabet_size=cnn_dim)

        cl.load_state_dict(sd['classifier'])
        fe.load_state_dict(sd['feature_extractor'])

        model = OmnilibStabilityPredictor(
            feature_extractor=fe,
            classifier=cl,
            alphabet=alphabet
        )

    else:
        raise ValueError(f"Model type {model_type} not recognized")
    
    return model.eval().to(device)

def test_model(
        model: torch.nn.Module,
        test_data: Union[str, pd.DataFrame],
        device: torch.device,
        pad_internally: bool=True,
        batch_size: int =256) -> Tuple[float, np.ndarray, np.ndarray]:
    
    '''Test a model's classification performance on a test dataset.
    Ensure that the dataset is already balanced for the AUC to be meaningful'''
    
    if isinstance(test_data, str):
        test_data = pd.read_csv(test_data)

    if pad_internally:
        test_data['cdr2_padded'] = test_data['CDR2'].apply(pad_cdr2)
        test_data['cdr3_padded'] = test_data['CDR3'].apply(lambda x: pad_internal(x, 28))
        test_data['seq'] = test_data['CDR1'] + test_data['cdr2_padded'] + test_data['cdr3_padded']
        sequences = list(test_data['seq'].values)

    else:
        test_data['seq'] = test_data['CDR1'] + test_data['CDR2'] + test_data['CDR3']
        sequences = [s.ljust(48, '-') for s in test_data['seq'].values]

    numseqs = torch.vstack([torch.tensor([AA2INDEX[aa] for aa in seq], dtype=torch.long) for seq in sequences])
    preds = []
    model.eval()
    for i in range(0, numseqs.shape[0], batch_size):
        batch = numseqs[i:i+batch_size].to(device)
        with torch.no_grad():
            preds.append(model.predict(batch, return_logits=True))
    all_preds = np.vstack(preds)

    labels = np.array(test_data['stability'] == 'high', dtype=int)
    auc = roc_auc_score(labels, all_preds)
    fpr, tpr, _ = roc_curve(labels, all_preds)

    return auc, fpr, tpr

def dms_redesign(
        cnnmodel: OmnilibStabilityPredictor,
        sequence: torch.Tensor,
        device: torch.device,
        batch_size: int = 21,
) -> torch.Tensor:
    

    mutated_sequences = []
    for i in range(sequence.shape[1]):
        for j in range(len(AA2INDEX)):
            mut = sequence.clone()
            mut[0, i] = j
            mutated_sequences.append(mut)
    mutated_sequences = torch.cat(mutated_sequences, dim=0)
    
    preds = []
    for i in range(0, len(mutated_sequences), batch_size):
        batch = mutated_sequences[i:i+batch_size].to(device)
        with torch.no_grad():
            preds.append(cnnmodel.predict(batch, return_logits = True))

    return np.stack(preds).squeeze()

def plot_scores_lr(lrmodel: OmnilibStabilityPredictor, 
                   sequence_tensors: List[torch.tensor], 
                   df: pd.DataFrame, 
                   unseen_weights: torch.tensor,
                   names: List[str], 
                   filename: str) -> None:
    

    fig, axs = plt.subplots(len(sequence_tensors), 1, figsize=(15, 10*len(sequence_tensors)))

    sns.set_theme(style='white', font_scale=1.0)

    cmap = mpl.colormaps.get_cmap('vlag_r')
    cmap.set_bad(color='gray')

    lr_weights = lrmodel.classifier.linear.weight.detach().cpu().reshape(21, -1)

    for i, seq_tensor in enumerate(sequence_tensors):

        seqscore = torch.gather(lr_weights, 0, seq_tensor)
        relative_score = lr_weights - seqscore
        ax = axs[i]
        g = sns.heatmap(relative_score, ax = ax, cmap=cmap, center=0, vmin = -1, vmax=1, cbar=False, \
            xticklabels = df['padded_sequence'].values[i], yticklabels = AA2INDEX.keys(), annot=True, \
                fmt=".2f", annot_kws = {'rotation': 90}, mask = unseen_weights.numpy())

        for j in range(seq_tensor.shape[1]):
            g.add_patch(plt.Rectangle((j, seq_tensor[0,j]), 1, 1, fill=False, edgecolor='black', lw=2))

        g.set_title(f'Difference in Predicted Stability to {names[i]} (using LR) - (Score {df["LR"].values[i]:.3f})')

    fig.tight_layout()

    if not isinstance(filename, str):
        filename = str(filename)

    if filename.endswith('.png'):
        plt.savefig(filename, dpi=300)
    else:
        plt.savefig(filename)


def plot_scores_cnn(
        base_scores: List[torch.tensor],
        dms_scores: List[torch.tensor],
        sequence_tensors: List[torch.tensor],
        df: pd.DataFrame,
        unseen_weights: torch.tensor,
        names: List[str],
        filename):

    fig, axs = plt.subplots(len(base_scores), 1, figsize=(15, 10*len(base_scores)))

    sns.set_theme(style='white', font_scale=1.0)

    cmap = mpl.colormaps.get_cmap('vlag_r')
    cmap.set_bad(color='gray')


    for i, (base, dms) in enumerate(zip(base_scores, dms_scores)):
        sequence_tensor = sequence_tensors[i]
        diffs = dms - base
        ax = axs[i]
        g = sns.heatmap(diffs.T, ax = ax, cmap=cmap, center=0, vmin = -1, vmax=1, cbar=False, \
            xticklabels = df['padded_sequence'].values[i], yticklabels = AA2INDEX.keys(), annot=True, \
                fmt=".2f", annot_kws = {'rotation': 90}, mask = unseen_weights.numpy())

        for j in range(sequence_tensor.shape[1]):
            g.add_patch(plt.Rectangle((j, sequence_tensor[0,j]), 1, 1, fill=False, edgecolor='black', lw=2))

        g.set_title(f'Difference in Predicted Stability to {names[i]} (using CNN) - (Score {df["CNN"].values[i]:.3f})')

    fig.tight_layout()

    if not isinstance(filename, str):
        filename = str(filename)

    if filename.endswith('.png'):
        plt.savefig(filename, dpi=300)
    else:
        plt.savefig(filename)