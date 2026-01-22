import os, gc
import platform
import logging
from datetime import datetime
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


def write_logs(logger, cfg, actual_dtype, config,n_trainable ,n_params, DEVICE, OUTPUT_PATH, checkpoint_path, gpu_id):
        
        logger.info("=" * 80)
        logger.info("*** General Info ***")
        logger.info(f"Python: {platform.python_version()}, PyTorch: {torch.__version__}")
        logger.info(f"OS: {platform.system()} {platform.release()}")
        logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Output Path: {OUTPUT_PATH}")
        logger.info(f"Training Type: {cfg.train.training_type}")
        logger.info(f"Epochs: {cfg.train.epoch}, Batch size: {cfg.data.batch_size}, Learning Rate: {cfg.train.lr}")
        logger.info(f"Eval Every: {cfg.train.eval_steps} Steps, Eval Sample: {cfg.train.eval_sample}")
        logger.info(f"Num Workers: {cfg.data.num_workers}, Pin Memory: {cfg.data.pin_memory}")
        logger.info(f"Shuffle: {cfg.data.shuffle}, Drop Last: {cfg.data.drop_last}")
        logger.info(f"Tokenizer: {cfg.data.hf_tokenizer}")
        logger.info(f"Checkpoint Path: {checkpoint_path}")
        logger.info(f"Dataset Source: {cfg.data.pretraining_hf_data if cfg.train.training_type == 'pre-train' else cfg.data.it_hf_data}")
        logger.info("")
        logger.info("*** Model Info ***")
        logger.info(f"Model: {cfg.train.model}")
        logger.info(f"Config Name: {config.config_name}")
        logger.info(f"Dtype: {actual_dtype}")
        logger.info(f"Trainable Params: {n_trainable/1e6:.2f}M / Total: {n_params/1e6:.2f}M")
        
        logger.info(f"Vocab Size: {cfg.model.vocab_size}")
        logger.info(f"Intermediate Size: {cfg.model.intermediate_size}")
        logger.info(f"Head Dim : {cfg.model.head_dim}")
        logger.info(f"Hidden Size: {cfg.model.hidden_size}")
        logger.info(f"Number of Attention Heads: {cfg.model.num_attention_heads}")
        logger.info(f"Number of Hidden Layers: {cfg.model.num_hidden_layers}")
        logger.info(f"Number of KV heads: {cfg.model.num_key_value_heads}")
        logger.info(f"Attention Type: {cfg.model.attn_type}")
        logger.info(f"Torch Dtype: {cfg.model.torch_dtype}")

        logger.info(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)