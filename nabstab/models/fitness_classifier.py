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
    
class ConvNet2(nn.Module):
    '''Variant of ConvNet with the MaskedPooling layer, that ignores padding,
    Otherwise, identical, with two convolutional layers and ReLU activations'''
    def __init__(self, alphabet_size, dim, ks1, ks2, pad_idx, pool_type='max'):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.pad_idx = pad_idx
        self.padding = ProperPadding()

        self.conv1 = nn.Conv1d(alphabet_size, dim, kernel_size=ks1, padding='same')
        self.conv2 = nn.Conv1d(dim, dim, ks2, padding='same')
        self.pooler = MaskedPooling(pool_type = pool_type)

    def forward(self, x):
        x_padding_mask = x.argmax(dim = 1).ne(self.pad_idx) #channel dimension is 1
        
        x = self.padding(x, x_padding_mask)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.padding(x, x_padding_mask)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.padding(x, x_padding_mask)
        x = self.pooler(x, x_padding_mask)
        
        return x
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
    
class MaskedPooling(nn.Module):
    """
    Pooling layer that respects padding masks.
    Takes a tensor of shape (batch_size, hidden_dim, seq_length) and a padding mask
    of shape (batch_size, seq_length), returns tensor of shape (batch_size, hidden_dim)
    containing the pooled values over non-padded positions.
    
    This follows the standard PyTorch convolution format where channels/features come
    before the sequence length dimension.
    
    Args:
        pool_type: str, either 'max' or 'avg' to specify the pooling operation
    """
    def __init__(self, pool_type='max'):
        super().__init__()
        assert pool_type in ['max', 'avg'], "pool_type must be either 'max' or 'avg'"
        self.pool_type = pool_type
    
    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, hidden_dim, seq_length)
            pad_mask: Boolean tensor of shape (batch_size, seq_length) where True indicates
                     valid positions and False indicates padded positions
        
        Returns:
            Tensor of shape (batch_size, hidden_dim) containing pooled values over non-padded positions
        """
        # Ensure proper dimensions
        assert x.dim() == 3, f"Expected 3D input tensor, got shape {x.shape}"
        assert pad_mask.dim() == 2, f"Expected 2D mask tensor, got shape {pad_mask.shape}"
        assert x.size(0) == pad_mask.size(0), "Batch sizes don't match"
        assert x.size(2) == pad_mask.size(1), "Sequence lengths don't match"
        
        # Expand mask to match input dimensions (B, 1, L) -> (B, d, L)
        mask_expanded = pad_mask.unsqueeze(1).expand(-1, x.size(1), -1)
        
        if self.pool_type == 'max':
            # For max pooling, mask padded positions with negative infinity
            masked_x = torch.where(mask_expanded, x, torch.tensor(float('-inf')))
            pooled, _ = torch.max(masked_x, dim=2)
        else:  # avg pooling
            # For average pooling, sum up non-padded values and divide by count
            masked_x = torch.where(mask_expanded, x, torch.zeros_like(x))
            # Sum along sequence length dimension
            summed = torch.sum(masked_x, dim=2)
            # Count non-padded elements per batch and feature
            counts = torch.sum(mask_expanded, dim=2).clamp(min=1)  # Avoid division by zero
            # Compute average
            pooled = summed / counts
        
        return pooled
    
class ProperPadding(nn.Module):
    """
    Module that ensures proper zero padding for sequence data before convolutions.
    Zeroes out all feature dimensions at pad positions to prevent information leakage
    during convolution operations.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, hidden_dim, seq_length)
            pad_mask: Boolean tensor of shape (batch_size, seq_length) where True indicates
                     valid positions and False indicates padded positions
        
        Returns:
            Tensor of same shape as input but with all feature dimensions zeroed out
            at pad positions
        """
        # Ensure proper dimensions
        assert x.dim() == 3, f"Expected 3D input tensor, got shape {x.shape}"
        assert pad_mask.dim() == 2, f"Expected 2D mask tensor, got shape {pad_mask.shape}"
        assert x.size(0) == pad_mask.size(0), "Batch sizes don't match"
        assert x.size(2) == pad_mask.size(1), "Sequence lengths don't match"
        
        # Expand mask to match input dimensions (B, 1, L) -> (B, d, L)
        mask_expanded = pad_mask.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Zero out all feature dimensions at pad positions
        return torch.where(mask_expanded, x, torch.zeros_like(x))

def gelu(x):
    """
    !!This is copied from ESM: github.com/facebookresearch/esm/blob/master/esm/modules.py!!
    Implementation of the gelu activation function. 

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LMHead(nn.Module):
    '''
    Projects hidden states to the vocabulary size for language modeling.
    Optionally accepts a weight matrix to use for the projection. This is typically tied to the input embeddings.
    '''
    def __init__(self, hidden_size, output_size, weight = None):
        super(LMHead, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        if weight is not None:
            self.weight = weight
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc(x)
        x = gelu(x)
        if hasattr(self, 'weight'):
            x = F.linear(x, self.weight) + self.bias
        else:
            x = self.fc2(x)
        return x
    
class StabilityPredictionHead(nn.Module):
    '''
    Projects hidden states to a single output for stability prediction.
    '''
    def __init__(self, rnn_hidden_size, hidden_dim=8):
        super(StabilityPredictionHead, self).__init__()
        self.fc1 = nn.Linear(rnn_hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class ProteinRNN(nn.Module):
    '''
    A simple RNN model for nanobody sequence modeling
    '''
    def __init__(self, input_size, hidden_size, rnn_type, alphabet):
        super(ProteinRNN, self).__init__()

        #General Params
        self.alphabet = alphabet
        self.pad_token = self.alphabet.get('-')
        self.hidden_size = hidden_size

        #Embedding Layer
        self.embedding = nn.Embedding(input_size, hidden_size)

        #RNN Component
        if rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError('Invalid RNN type')

        #Output heads
        self.lm_head = LMHead(hidden_size, input_size, self.embedding.weight) #tied weights to project back
        self.classification_head = StabilityPredictionHead(hidden_size)

        #loss functions
        self.lm_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token, reduction='mean') #ignore gaps
        self.classification_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        return output, hidden
    
    def lm_training_step(self, batch):
        '''
        Trains the language model head. Runs the RNN on the input and feeds the output to the language model head.
        The targets are the input sequence shifted by one (next token prediction)
        '''
        input, y = batch
        output, _ = self(input)
        output_logits = self.lm_head(output)
        targets = input.clone()
        targets[:, :-1] = input[:, 1:]
        lm_loss = self.lm_loss_fn(output_logits.transpose(1,2), targets)
        #some other metrics
        ppl = torch.exp(lm_loss) #perplexity
        recon = output_logits.argmax(-1) #reconstructed sequence
        mask = (targets != self.pad_token)
        acc = (recon == targets).float().masked_select(mask).mean() #only calculate accuracy on non-pad tokens (where loss is calculated)
        return {'lm_loss': lm_loss, 'ppl': ppl, 'lm_acc': acc}
    
    def classification_training_step(self, batch):
        '''
        Trains the stability prediction head. Runs the RNN on the input and averages the hidden states of the non-pad tokens, which are then fed to the stability prediction head.
        '''
        input, y = batch
        output, _ = self(input) #output is (batch, seq_len, hidden_size)
        #mask out the pad tokens - only average the hidden states of the non-pad tokens
        mask = (input != self.pad_token)
        denom = mask.sum(-1, keepdim=True)
        mean_feat = torch.sum(output * mask.view(*mask.shape, 1), dim=1) / denom #(batch, hidden_size)

        stability_logits = self.classification_head(mean_feat)
        stability_loss = self.classification_loss_fn(stability_logits, y.view(-1, 1))
        acc = ((stability_logits > 0) == y.view(-1, 1)).float().mean() #what a shitty way to calculate accuracy
        return {'classification_loss': stability_loss, 'classification_acc': acc}
    
    def combined_training_step(self, batch):
        lm_metrics = self.lm_training_step(batch)
        classification_metrics = self.classification_training_step(batch)
        loss = lm_metrics['lm_loss'] + classification_metrics['classification_loss']
        metrics = {**lm_metrics, **classification_metrics, 'loss': loss}
        return metrics
    
    def predict(self, input, return_logits=False):
        '''
        Predicts the stability of a sequence.
        input: a tensor of shape (batch, seq_len)
        '''
        if input.ndim == 1:
            #single input - add batch dimension
            input = input.view(1, -1)

        output, _ = self(input)
        mask = (input != self.pad_token)
        denom = mask.sum(-1, keepdim=True)
        mean_feat = torch.sum(output * mask.view(*mask.shape, 1), dim=1) / denom #(batch, hidden_size)
        stability_logits = self.classification_head(mean_feat)

        if return_logits:
            return stability_logits.detach().cpu().numpy()
        else:
            return torch.sigmoid(stability_logits).detach().cpu().numpy()

    def sample(self, start_token, max_length=48):
        generated_seq = []
        output, hidden = self(start_token)
        output_probs = F.softmax(output[:, 0], dim=1)
        start_token = torch.multinomial(output_probs, 1)
        for i in range(max_length-1):
            output, hidden = self(start_token, hidden[i])
            probs = F.softmax(output[:, i+1], dim=1)
            start_token = torch.multinomial(probs, 1)
            generated_seq.append(start_token.item())
        return generated_seq
