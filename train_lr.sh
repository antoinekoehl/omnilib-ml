#!/bin/bash

# This script is used to train a model using the omnilib-ml library.

python train_model.py \
    --train_data_path "data/model_training_data/ENN_train.csv" \
    --test_data_path "data/model_training_data/ENN_test.csv" \
    --model_type "lr" \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --device "cpu" \
    --save_model_path "data/trained_models/"