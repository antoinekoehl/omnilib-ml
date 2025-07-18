import os
import argparse
from datetime import datetime
from functools import partial
import multiprocessing as mp
import logging

from tqdm import tqdm
import pandas as pd

from nabstab.tango import (
    NanobodyAggregationPredictor
)

def setup_logger(output_dir):
    """Set up logger to write to both console and file."""
    # Create logs directory inside output directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"tango_prediction_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger("tango_prediction")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def process_row(row, predictor, logger):
    """Process a single row from the dataframe."""
    cdr1 = row.CDR1
    cdr2 = row.CDR2
    cdr3 = row.CDR3
    stability = row.stability
    
    try:
        result = predictor.predict(
            cdr1=cdr1,
            cdr2=cdr2,
            cdr3=cdr3
        )
        if result is not None:
            max_agg = result['max_agg']
            mean_agg = result['mean_agg']
            agg_regions = len(result['agg_regions'])
            total_score = result['total_agg_score']
            normalized_score = result['normalized_agg_score']
            return [max_agg, mean_agg, agg_regions, total_score, normalized_score, stability]
    except Exception as e:
        logger.error(f"Error processing sequence {row.name}: {cdr1}, {cdr2}, {cdr3} - {str(e)}")
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tango prediction for nanobody aggregation.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV file containing nanobody sequences."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of samples from each class to predict (default: 50000)"
    )
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)

    logger.info("Loading Data")
    # Load input data, subset
    input_df = pd.read_csv(args.input_file)

    input_df = pd.concat(
        [
            input_df[
                input_df.stability == 'high'
                ].sample(args.num_samples, random_state=42),
            input_df[
                input_df.stability == 'low'
                ].sample(args.num_samples, random_state=42)
        ]
    )

    logger.info(f"Loaded {len(input_df)} sequences")

    tango_executable = "tango_x86_64_release"
    predictor = NanobodyAggregationPredictor(
        tango_executable=tango_executable,
        output_dir=args.output_dir,
    )

    logger.info("Starting Tango predictions")
    all_results = []

    for i, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing sequences"):
        result = process_row(row, predictor, logger)
        all_results.append(result)

    # Filter out None results and create dataframe
    new_data = [r for r in all_results if r is not None]
    
    logger.info(f"Successfully processed {len(new_data)} out of {len(input_df)} sequences")
    
    # Create output dataframe
    tango_df = pd.DataFrame(
        new_data,
        columns=[
            'max_agg',
            'mean_agg',
            'agg_regions',
            'total_agg_score',
            'normalized_agg_score',
            'stability'
        ]
    )
    
    # Save results
    output_path = os.path.join(args.output_dir, "tango_results.csv")
    tango_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
