import os
import logging
from typing import Optional
import time

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from nabstab.models import (
    OmnilibStabilityPredictor,
    LinearNet,
    ConvNet,
    FC
)
from nabstab.datasets.classifier_dataset import NbStabilityDataset
from nabstab.constants import AA2INDEX
from nabstab.utils import train_model


def create_lr_model(sequence_length: int, alphabet_size: int = len(AA2INDEX)):
    """
    Create a logistic regression model for classifying nanobody stability.
    """
    fe = nn.Identity() # No special feature extraction
    cl = LinearNet(
        sequence_length=sequence_length,
        alphabet_size=alphabet_size
    )

    model = OmnilibStabilityPredictor(
        feature_extractor=fe,
        classifier=cl,
        alphabet=AA2INDEX
    )

    return model

def create_cnn_model(
        alphabet_size: int = len(AA2INDEX),
        dim: int = 24,
        ks1: int = 5,
        ks2: int = 9,
        h_dim: Optional[int] = 8
):
    """
    Create a CNN model for classifying nanobody stability.

    Args:
        alphabet_size (int): Size of the alphabet (default is 21 for amino acids + GAP).
        dim (int): Dimension of the convolutional layers.
        ks1 (int): Kernel size for the first convolutional layer.
        ks2 (int): Kernel size for the second convolutional layer.
        h_dim (Optional[int]): Hidden dimension for the fully connected layer. If None, no hidden layer is used.
    """
    
    fe = ConvNet(
        alphabet_size=alphabet_size,
        dim=dim,
        ks1=ks1,
        ks2=ks2
    )

    if h_dim is not None:
        cl = FC(
            alphabet_size=dim,
            sequence_length=1, # pooled
            h_dim=h_dim,
            out_size=1 # output size for binary classification
        )
    else:
        cl = LinearNet(
            sequence_length=1, # pooled
            alphabet_size=dim
        )

    model = OmnilibStabilityPredictor(
        feature_extractor=fe,
        classifier=cl,
        alphabet=AA2INDEX
    )

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model for nanobody stability prediction.")
    parser.add_argument("--model_type", type=str, choices=["lr", "cnn"], required=True,
                        help="Type of model to train: 'lr' for logistic regression, 'cnn' for convolutional neural network.")
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the training data CSV file.")
    parser.add_argument("--test_data_path", type=str, required=False,
                        help="Path to the test data CSV file.")
    parser.add_argument("--ks1", type=int, default=5,
                        help="Kernel size for the first convolutional layer (only for CNN).")
    parser.add_argument("--ks2", type=int, default=9,
                        help="Kernel size for the second convolutional layer (only for CNN).")
    parser.add_argument("--dim", type=int, default=24,
                        help="Dimension of the convolutional layers (only for CNN).")
    parser.add_argument("--h_dim", type=int, default=8,
                        help="Hidden dimension for the fully connected layer (only for CNN).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and validation.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs to train the model.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for the optimizer.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the training on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--save_model_path", type=str, default="model.pth",
                        help="Path to save the trained model.")
    args = parser.parse_args()

    #setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Training on device: {args.device}")

    # Load training and validation datasets
    train_df = pd.read_csv(args.train_data_path)
    if args.test_data_path:
        test_df = pd.read_csv(args.test_data_path)
    else:
        test_df = None
    
    ds = NbStabilityDataset(
        df=train_df,
        alphabet=AA2INDEX,
        cdr3_max_len=28 #what we use in the paper
    )

    train_ds, val_ds = random_split(ds, [0.8, 0.2]) #80/20 split for train/val
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    logging.info(f"Loaded {len(train_ds)} training samples")
    logging.info(f"Loaded {len(val_ds)} validation samples")

    # Create model
    if args.model_type == "lr":
        model = create_lr_model(sequence_length=ds.sequence_length, alphabet_size=len(AA2INDEX))
    elif args.model_type == "cnn":
        model = create_cnn_model(
            alphabet_size=len(AA2INDEX),
            dim=args.dim,
            ks1=args.ks1,
            ks2=args.ks2,
            h_dim=args.h_dim
        )
    else:
        raise ValueError(f"Model type {args.model_type} not recognized. Use 'lr' or 'cnn'.")
    
    model.to(args.device)
    optimizer = model.configure_optimizers(lr=args.learning_rate, wd=args.weight_decay)

    logging.info(f"Model created: {args.model_type} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    start_time = time.time()

    # Train the model
    final_val_loss, final_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        optimizer=optimizer,
        epochs=args.num_epochs
    )

    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

    #save the model
    if args.model_type == "lr":
        save_dict = {
            'model_state_dict': model.state_dict(),
            'params': {
                'sequence_length': ds.sequence_length,
                'alphabet_size': len(AA2INDEX)
            }
        }
        model_save_name = 'lr.pth'
    elif args.model_type == "cnn":
        params = {
            'dim': args.dim,
            'ks1': args.ks1,
            'ks2': args.ks2,
            'alphabet_size': len(AA2INDEX),
            'h_dim': args.h_dim
        }
        save_dict = {
            'params': params,
            'alphabet': AA2INDEX,
            'feature_extractor': model.feature_extractor.state_dict(),
            'classifier': model.classifier.state_dict(),
        }
        model_save_name = 'cnn.pth'
    else:
        raise ValueError(f"Model type {args.model_type} not recognized. Use 'lr' or 'cnn'.")

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    output_file = os.path.join(args.save_model_path, model_save_name)
    torch.save(save_dict, output_file)

    logging.info(f"Model saved to {output_file}")
    logging.info(f"Final validation loss: {final_val_loss:.4f}, accuracy: {final_val_acc:.4f}")
    logging.info("Training complete.")
