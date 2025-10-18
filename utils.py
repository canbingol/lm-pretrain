import os, gc
import argparse
import yaml
from dataclasses import dataclass

import torch
import sentencepiece as spm

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

def get_tokenizer(hf_data,text_column_name,output_path,vocab_size,train_tokenizer,gpu_id):
    from data_prepare import prepare_tokenizer_data

    tokenizer_data_path = prepare_tokenizer_data(hf_data,text_column_name,output_path,gpu_id)
    tokenizer_path = f"{output_path}/{os.path.basename(output_path)}.tokenizer.model"
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
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--training-type", type=str, help="training type")

    parser.add_argument('--model', type=str, choices=["qwen3","deepseek","gemma2","llama2"])
    parser.add_argument('--epoch', type=int)
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

def format_it_data(query, input, answer):
    query_str = f"<|user|>{query}<|user_end|>\n"
    input_str = f"<|input|>{input}<|input_end|>\n" if input else ""
    answer_str = f"<|assistant|>{answer}<|assistant_end|>"

    return query_str + input_str + answer_str
