import os, gc
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

def load_model(model,inference,checkpoint_path, device,optimizer=None,scheduler=None):
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if inference:
        return model.to(device)

    train_loss  = checkpoint["train_loss"]
    val_loss    = checkpoint["val_loss"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Checkpoint train loss : {train_loss} | validation loss : {val_loss}")
    return model, optimizer,scheduler

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_argparser():
    parser = argparse.ArgumentParser()
    # model choosing
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--training-type", type=str, help="training type")

    parser.add_argument('--model', type=str, choices=["qwen3", "deepseek", "gemma2", "llama2", "gptoss", "gpt2"])
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--world-size', type=int)

    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--max-seq-len', type=int)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--training-steps', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--eval-steps', type=int)
    parser.add_argument('--eval-sample', type=int)
    parser.add_argument('--vocab-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--text-column-name', type=str)
    parser.add_argument('--model-info', action="store_true")
    parser.add_argument('--train-tokenizer', action="store_true")
    parser.add_argument('--inference', action="store_true")
    parser.add_argument('--force', action="store_true")
    parser.add_argument('--pre-training-hf-data', type=str)
    parser.add_argument('--it-hf-data', type=str)
    parser.add_argument('--hf-tokenizer', type=str)

    parser.add_argument('--world_size', type=int)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--saved-token-path', type=str)

    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--drop-last', action='store_true', default=True)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--pin-memory', action='store_true', default=True)
    parser.add_argument('--single-file', default=None, type=str)


    return parser

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_args_with_yaml(args,yaml_cfg):
    for key, value in yaml_cfg.items():

        key_ = key.replace("-","_")
        if getattr(args, key_,None) in [None, False]:
            setattr(args, key_, value)
    return args

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