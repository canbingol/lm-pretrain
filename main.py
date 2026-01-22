# stdlib
import os
from datetime import datetime

# third-party
import torch
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer
import numpy as np

from huggingface_hub import login, HfApi

# local modules - top-level logic
from train.trainer import Trainer
from data_prepare import (
    prepare_pretrain_data,
    create_tokens_file,
    prepare_it_data,
)
from utils import (
    write_logs,
    TrainState,
    TrainConfig,
    DataLoaders,
    setup_logger,
    get_unique_filename,
)

from configs.loader import set_config, get_config

# model registry - separated by domain
from models.decoder_model import DecoderCausalLM, ModelConfig

# Seed info
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

def ddp_setup(WORLD_SIZE):
    
    backend = "gloo" if WORLD_SIZE < 2 else "nccl"

    init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_pretrain_data(gpu_id, TOKEN_PATH, BATCH_SIZE, SHUFFLE,
                        DROP_LAST, NUM_WORKERS, PIN_MEMORY, SINGLE_FILE, logger):
    """
    This Function create train and validation loaders for pre training
    """
    train_loader, val_loader = prepare_pretrain_data(
        token_file_data_dir=TOKEN_PATH, batch_size=BATCH_SIZE,shuffle=SHUFFLE,
        drop_last=DROP_LAST, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, single_file=SINGLE_FILE,
        gpu_id=gpu_id
        )
    if gpu_id == 0:
        logger.info(f"Train-loader size: {len(train_loader)}, Val-loader size: {len(val_loader)} ")

    return train_loader, val_loader

def get_it_data(gpu_id, tokenizer, IT_HF_DATA, BATCH_SIZE, MAX_SEQ_LEN):
    """
    This Function create train and validation loaders for instruction tuning
    """
    train_loader, val_loader = prepare_it_data(
        hf_dataset=IT_HF_DATA, tokenizer=tokenizer, batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN, pad_token=tokenizer.pad_token_id, gpu_id=gpu_id
    )

    return train_loader, val_loader

def main():

    cfg = get_config()

    checkpoint_path = cfg.train.checkpoint

    ddp_setup(cfg.train.world_size)

    TOKEN_PATH = None
    if cfg.train.training_type == "pre-train":
        TOKEN_PATH = str(cfg.data.token_path)

    # If given saved token path not exist create .bin files with given hf dataset
    if cfg.train.training_type != "pre-train" or TOKEN_PATH is None or not os.path.exists(TOKEN_PATH) or len(os.listdir(TOKEN_PATH)) == 0:
        TOKEN_PATH = create_tokens_file(hf_dataset=cfg.data.pretraining_hf_data, hf_tokenizer=cfg.data.hf_tokenizer, base_dir=TOKEN_PATH,
                                               test_split=cfg.data.test_split, tokens_chunks_size=cfg.data.tokens_chunks_size, dtype=cfg.data.token_dtype)


    model_map = {
        "decoder": (ModelConfig, DecoderCausalLM)
    }

    dtype_map = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "float32": torch.float32,  
    "bfloat16": torch.bfloat16
}

    ConfigClass, ModelClass = model_map[cfg.train.model]
    config = ConfigClass()
    set_config(config, cfg.model)
    actual_dtype = dtype_map.get(config.torch_dtype, torch.float32)

    gpu_id = int(os.environ["LOCAL_RANK"])
    DEVICE = torch.device(f"cuda:{gpu_id}")
    config.device = DEVICE

    log_name = f"train_{cfg.train.model}_{cfg.train.training_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    model = ModelClass(config).to(DEVICE)
    model = model.to(actual_dtype)   

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if cfg.train.force:
        OUTPUT_PATH = f"output/{cfg.train.model}/{cfg.train.training_type}/{n_params/1e6:.2f}M_force"
    else:
        OUTPUT_PATH = f"output/{cfg.train.model}_{cfg.train.training_type}_{n_params/1e6:.2f}M"

    OUTPUT_PATH = get_unique_filename(OUTPUT_PATH)

    logger = setup_logger(rank=gpu_id, log_dir=OUTPUT_PATH, filename=log_name)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.hf_tokenizer)


    if gpu_id == 0:
        write_logs(logger, cfg, actual_dtype, config,n_trainable ,n_params, DEVICE, OUTPUT_PATH, checkpoint_path, gpu_id)


    # Prepare training data for pre-train or instruction tuning
    if cfg.train.training_type == "pre-train":
        train_loader, val_loader = get_pretrain_data(gpu_id, TOKEN_PATH, cfg.data.batch_size, cfg.data.shuffle,
                        cfg.data.drop_last, cfg.data.num_workers, cfg.data.pin_memory, False, logger)

    elif cfg.train.training_type == "instruction-tuning":
        train_loader, val_loader = get_it_data(gpu_id, tokenizer, cfg.data.it_hf_data, cfg.data.batch_size, cfg.data.max_seq_len)

    else:
        print(f"Invalid cfg.train.training_type argument: {cfg.train.training_type}. Valid options are: pre-train, instruction-tuning.")

    TRAINING_STEPS = cfg.train.training_steps

    TRAINING_STEPS = len(train_loader) if not TRAINING_STEPS else TRAINING_STEPS
    logger.info(f"TRAINING_STEPS : {TRAINING_STEPS}")
    logger.info(f"train loader lenght: {len(train_loader)}")
    logger.info(f"validaiton loader lenght: {len(val_loader)}")

    # Create arguments for training
    data_loaders = DataLoaders(
        train=train_loader,
        val=val_loader
    )

    train_state = TrainState(
        model=model,
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer
    )

    train_config = TrainConfig(
        num_epochs=cfg.train.epoch,
        training_steps=TRAINING_STEPS,
        eval_steps=cfg.train.eval_steps,
        eval_sample=cfg.train.eval_sample,
        learning_rate=cfg.train.lr,
        device=gpu_id,
        output_path=OUTPUT_PATH,
        force=cfg.train.force,
    )

    # Start training
    try:
        trainer = Trainer(train_state,data_loaders,train_config, logger)
        trainer.train()

        if cfg.hub.push_to_hub:
            api = HfApi()

            model.push_to_hub(cfg.hub.repo_name)
            tokenizer.push_to_hub(cfg.hub.repo_name)
            
            api.upload_folder(
                folder_path=OUTPUT_PATH,          
                path_in_repo="logs_and_best-model",         
                repo_id=cfg.hub.repo_name,           
                repo_type="model",                
            )
            api.upload_file(
                path_or_fileobj="./config/setup_config.yaml",
                path_in_repo="setup_config.yaml",
                repo_id=cfg.hub.repo_name,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj="./models/decoder_model.py",
                path_in_repo="model.py",
                repo_id=cfg.hub.repo_name,
                repo_type="model",
            )
            
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        raise
    finally:
        destroy_process_group()

if __name__ == "__main__":
    main()
