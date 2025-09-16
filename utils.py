import os, gc
import argparse
from dataclasses import dataclass

import torch
import sentencepiece as spm

from data_loader import prepare_tokenizer_data
from train_tokenizer import TrainTokenizer

from dataclasses import dataclass
from typing import Optional
import torch
from torch.utils.data import DataLoader

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
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    checkpoint_path: str

def load_model(model,inference,checkpoint_path, device,optimizer=None,scheduler=None):
    checkpoint = torch.load(checkpoint_path,map_location=device)
    if inference:
        return model.to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    train_loss  = checkpoint["train_loss"]
    val_loss    = checkpoint["val_loss"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    print(f"Checkpoint train loss : {train_loss} | validation loss : {val_loss}")
    return model, optimizer,scheduler

def get_tokenizer(hf_data,text_column_name,output_path,vocab_size,train_tokenizer):
    tokenizer_data_path = prepare_tokenizer_data(hf_data,text_column_name,output_path)
    tokenizer_path = f"{output_path}/{output_path}.tokenizer.model"
    if train_tokenizer or not os.path.exists(tokenizer_path):
        tokenizer_path = TrainTokenizer(vocab_size,tokenizer_data_path,output_path).get_model_path()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer

def model_size_info(model):
    num_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    size_mb = param_size_bytes / (1024**2)
    size_gb = param_size_bytes / (1024**3)

    print(f"Number of total params: {num_params:,}")
    print(f"Model size: {size_mb:.2f} MB ({size_gb:.2f} GB)")


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def build_argparser():
    parser = argparse.ArgumentParser()
    # model choosing
    parser.add_argument('--model', type=str, required=True, choices=["qwen3"],
                        help='Model name to use (e.g., qwen3)')

    # Training arguments
    parser.add_argument('--epoch', type=int, default=1,
                        help='Number of training epochs')

    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Maximum number of tokens to generate during inference')

    parser.add_argument('--prompt', type=str, default="Merhaba ben ",
                        help='Prompt text used for inference')

    parser.add_argument('--training-steps', type=int, default=None,
                        help='Maximum number of training steps (if None, computed automatically)')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    parser.add_argument('--eval-steps', type=int, default=200,
                        help='Number of evaluation steps per evaluation run')
    
    parser.add_argument('--eval-sample', type=int, default=200,
                        help='Number of evaluation steps per evaluation run')

    parser.add_argument('--vocab-size', type=int, default=10_000,
                        help='Vocabulary size of the model/tokenizer')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training/evaluation')

    parser.add_argument('--text-column_name', type=str, default="text",
                        help='Name of the text column in the HF dataset')

    parser.add_argument('--model-info', action="store_true", default=False,
                        help='Print model configuration information')

    parser.add_argument('--train-tokenizer', action="store_true", default=False,
                        help='Enable training a new tokenizer')

    parser.add_argument('--inference', action="store_true", default=False,
                        help='Run inference instead of training')

    parser.add_argument('--force', action="store_true", default=False,
                        help='Start training without loading checkpoint')

    parser.add_argument('--hf-data', type=str, default=None,
                        help='Name or path of the Hugging Face dataset')

    parser.add_argument('--device', type=str, default="cuda",
                        help='Device')
    return parser