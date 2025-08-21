#!/bin/bash

python evaluate_fitness_scores.py \
    --fasta_file 'data/rescue_data/nbs_to_opt.fasta' \
    --model_paths "data/trained_models/cnn.pth" "data/trained_models/lr.pth" \
    --model_names "cnn" "lr" \
    --output_directory "data/evaluated_fitness_scores" \
    --batch_size 256 \
    --device "cpu"