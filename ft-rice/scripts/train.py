import os
import argparse
import json

import swanlab
import pandas as pandas
import torch
from functools import partial

# Huggingface transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments
)

# Local imports
from src.util import (
    dist_print,
    is_main_process,
    setup_distributed,
    setup_logging,
    setup_seed,
    get_index,
    setup_sync_batchnorm
)
from src.dataset import MultiTrackDataset
from src.model import GenOmics, targets_scaling_torch
from src.metrices import compute_multimodal_metrics
from src.trainer import (
    CustomTrainer,
    DistributedSamplerCallback,
    LocalLoggerCallback
)

def parse_args():
    """
    Parse CLI arguments and return an ``args`` object.
    """
    parser = argparse.ArgumentParser(description="Train RNA-seq track predictor with configurable args.")

    # --- data paths ---
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained model (hugging face format).")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Path to the tokenizer (hugging face format).")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Checkpoint directory to resume training from.")
    parser.add_argument("--sequence_split_train", type=str,
                        help="Training split index file.")
    parser.add_argument("--sequence_split_train_multi", type=str, nargs="+",
                        help="Training split index files (multiple).")
    parser.add_argument("--sequence_split_val", type=str, required=True,
                        help="Validation split index file.")
    parser.add_argument("--index_stat_json", type=str,
                        help="Training data statistics JSON (index_stat.json).")
    parser.add_argument("--index_stat_multi_json", type=str, nargs="+",
                        help="Multiple training data statistics JSONs (index_stat.json).")
    parser.add_argument("--nonzero_means", type=float, nars="+",
                        help="Per-track non-zero mean values.")

    # ---  Output settings ---
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Base output directory.")
    
    # --- Debugging / convenience ---
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Debug: limit number of training samples (None means no limit).")
    parser.add_argument("--max_sequence_length", type=int, default=32768)

    # --- Chromosome splits ---
    parser.add_argument("--train_chromosomes", type=str, nargs="+", default=["Chr19"],
                        help="List of chromosomes used for training.")
    parser.add_argument("--val_chromosomes", type=str, nargs="+", default=["Chr12"],
                        help="List of chromosomes used for validation.")
    
    # --- Training hyperparameters ---
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size_per_device", type=int, default=1, help="Per-GPU batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--gpus_per_node", type=int, default=8, help="Number of GPUs per node.")

    # --- model settings ---
    parser.add_argument("--loss_func", type=str, default="mse", choices=["mse", "poisson", "tweedie", "poisson-multinomial"],
                        help="Loss function type.")
    parser.add_argument("--proj_dim", type=int, default=1024, help="U-Net input feature dimension.")
    parser.add_argument("--num_downsamples", type=int, default=4, help="Number of downsampling blocks in the U-Net.")
    parser.add_argument("--bottlencek_dim", type=int, default=1536, help="U-Net bottleneck dimension.")

    # --- Misc ---
    parser.add_argument("--use_swanlab", action="store_true", help="Enable log into swanlab.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # configure
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Enable FlashAttention acceleration (default: disabled).")

    return parser.parse_args()

def main():
    """
    Main training entrypoint: fine-tune a pretrained DNA language modek with multi-track BigWig
    signals for single-base-resolution prediction.

    Supports Distributed Data Parallel (DDP), optional FlashAttention-2 + bf16 acceleration,
    and swanlab logging.
    """

def