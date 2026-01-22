import os, gc
import yaml
import logging
import argparse
import yaml
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

@dataclass(frozen=True)
class TrainConfig:
    num_epochs: int
    training_steps: int
    eval_steps: int
    eval_sample: int
    learning_rate: float
    device: torch.device
    output_path: str
    force: bool = False

@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader

@dataclass
class TrainState:
    model: torch.nn.Module
    checkpoint_path: str
    tokenizer: AutoModel

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def format_it_data(query: str, input: str, answer: str) -> str:
    """
    Formats instruction-tuning samples using Kumru-2B's native chat-style tokens.
    """

    # user message
    query_block = (
        f"<|start_header_id|>user<|end_header_id|>\n{query.strip()}\n"
    )

    # optional input
    input_block = (
        f"<|start_header_id|>input<|end_header_id|>\n{input.strip()}\n"
        if input
        else ""
    )

    # assistant answer
    answer_block = (
        f"<|start_header_id|>assistant<|end_header_id|>\n{answer.strip()}\n"
    )

    return query_block + input_block + answer_block


def setup_logger(rank=0, log_dir="logs", filename="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)

    handlers = [logging.FileHandler(log_file)]
    if rank == 0:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
    return logging.getLogger(__name__)


def get_unique_filename(path):
    counter = 1
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{path}_{counter}"
        counter += 1
    return new_path