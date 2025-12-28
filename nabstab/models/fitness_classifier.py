import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OmnilibStabilityPredictor(nn.Module):
    def __init__(self, alphabet, feature_extractor=None, classifier=None):
        super(OmnilibStabilityPredictor, self).__init__()

        self.alphabet = alphabet
        assert feature_extractor is not None and classifier is not None, "Feature extractor and classifier must be specified"
        self.feature_extractor = feature_extractor
        self.classifier = classifier

        self.loss_fn = nn.BCEWithLogitsLoss()

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):
        oh = F.one_hot(x, num_classes=len(self.alphabet)).float().transpose(1,2)
        emb = self.feature_extractor(oh)
        logits = self.classifier(emb)
        return logits
    
    def predict(self, x, return_logits=False):
        with torch.no_grad():
            logits = self.forward(x)
            if return_logits:
                return logits.cpu().numpy()
            else:
                return torch.sigmoid(logits).cpu().numpy()
            
    def classification_acc(self, logits, labels):
            preds = torch.round(torch.sigmoid(logits))
            if preds.shape != labels.shape:
                labels = labels.unsqueeze(-1)
            return (preds == labels).float().mean()
        
    def configure_optimizers(self, lr=1e-3, wd=0.0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        
   
'''
Defined subnetwork architectures that can be plugged in
'''
class ConvNet(nn.Module):
    def __init__(self, alphabet_size, dim, ks1, ks2):
        super().__init__()
        self.alphabet_size = alphabet_size

        self.conv1 = nn.Conv1d(alphabet_size, dim, kernel_size=ks1, padding='same')
        self.conv2 = nn.Conv1d(dim, dim, ks2, padding='same')
        self.pooler = nn.AdaptiveMaxPool1d(1)

        self.network = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.pooler,
        )

    def forward(self, x):
        return self.network(x)
    
class FC(nn.Module):
    '''Two layer MLP with ReLU'''
    def __init__(self, alphabet_size, sequence_length, h_dim, out_size):
        super().__init__()
        self.alphabet_size = alphabet_size

        self.fc1 = nn.Linear(alphabet_size * sequence_length, h_dim)
        self.fc2 = nn.Linear(h_dim, out_size)

        self.network = nn.Sequential(
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

    def forward(self, x):
        return self.network(x)
    
class LinearNet(nn.Module):
    '''Linear transform of a one-hot input to a scalar.
    This class is used for logistic regression'''
    def __init__(self, sequence_length, alphabet_size):
        super().__init__()
        self.alphabet_size = alphabet_size

        self.linear = nn.Linear(sequence_length * alphabet_size, 1)

        self.network = nn.Sequential(
            nn.Flatten(),
            self.linear
        )

    def forward(self, x):
        return self.network(x)