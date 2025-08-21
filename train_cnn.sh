#!/bin/bash

python train_model.py \
    --train_data_path "data/model_training_data/train_subset.csv" \
    --test_data_path "data/model_training_data/ENN_test.csv" \
    --model_type "cnn" \
    --ks1 5 \
    --ks2 9 \
    --dim 24 \
    --h_dim 8 \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --device "cpu" \
    --save_model_path "data/trained_models"