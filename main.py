# stdlib
import os
import platform
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
    get_config,
    TrainState,
    TrainConfig,
    DataLoaders,
    setup_logger,
    get_unique_filename
)

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

    config = get_config()

# config: YAML'dan load edilmiş dict

    WORLD_SIZE = int(config["train"]["world_size"])
    TRAINING_TYPE = str(config["train"]["training_type"])
    MODEL = str(config["train"]["model"])

    VOCAB_SIZE = int(config["data"]["vocab_size"])
    BATCH_SIZE = int(config["data"]["batch_size"])
    EPOCH = int(config["train"]["epoch"])

    EVAL_SAMPLE = int(config["train"]["eval_sample"])
    EVAL_STEPS = int(config["train"]["eval_steps"])
    LR = float(config["train"]["lr"])
    ATTN_TYPE = str(config["train"]["attn"])

    PUSH_TO_HUB = bool(config["hub"]["push_to_hub"])
    REPO_NAME = str(config["hub"]["repo_name"])
    SHUFFLE = bool(config["data"]["shuffle"])
    DROP_LAST = bool(config["data"]["drop_last"])
    NUM_WORKERS = int(config["data"]["num_workers"])
    PIN_MEMORY = bool(config["data"]["pin_memory"])

    SINGLE_FILE = str(config["data"]["single_file"]) if config["data"].get("single_file") is not None else None

    TEST_SPLIT = int(config["data"]["test_split"])
    TOKENS_CHUNK_SIZE = int(config["data"]["tokens_chunks_size"])
    TOKEN_DTYPE = str(config["data"]["token_dtype"])
    PRE_TRAINING_HF_DATA = str(config["data"]["pretraining_hf_data"])
    IT_HF_DATA = str(config["data"]["it_hf_data"])
    HF_TOKENIZER = str(config["data"]["hf_tokenizer"])

    FORCE = bool(config["train"]["force"])

    INFERENCE = bool(config["inference"]["inference"])
    PROMPT = str(config["inference"]["prompt"])
    MAX_NEW_TOKENS = int(config["inference"]["max_new_tokens"])

    MAX_SEQ_LEN = int(config["data"]["max_seq_len"])

    TRAINING_STEPS = config["train"].get("training_steps", None)
    TRAINING_STEPS = int(TRAINING_STEPS) if TRAINING_STEPS is not None else None
    
    # checkpoint_path: INFERENCE'a göre seç
    if INFERENCE:
        checkpoint_path = config["inference"].get("checkpoint", None)
    else:
        checkpoint_path = config["train"].get("checkpoint", None)

    checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None


    ddp_setup(WORLD_SIZE)

    TOKEN_PATH = None
    if TRAINING_TYPE == "pre-train":
        TOKEN_PATH = str(config["data"]["token_path"])

    # If given saved token path not exist create .bin files with given hf dataset
    if TRAINING_TYPE != "pre-train" or TOKEN_PATH is None or not os.path.exists(TOKEN_PATH) or len(os.listdir(TOKEN_PATH)) == 0:
        TOKEN_PATH = create_tokens_file(hf_dataset=PRE_TRAINING_HF_DATA, hf_tokenizer=HF_TOKENIZER, base_dir=TOKEN_PATH,
                                               test_split=TEST_SPLIT, tokens_chunks_size=TOKENS_CHUNK_SIZE, dtype=TOKEN_DTYPE)


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

    ConfigClass, ModelClass = model_map[MODEL]
    config = ConfigClass()
    actual_dtype = dtype_map.get(config.torch_dtype, torch.float32)

    config.vocab_size = VOCAB_SIZE
    config.batch_size = BATCH_SIZE
    config.attn_type = ATTN_TYPE

    gpu_id = int(os.environ["LOCAL_RANK"])
    DEVICE = torch.device(f"cuda:{gpu_id}")
    config.device = DEVICE
    log_name = f"train_{MODEL}_{TRAINING_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


    model = ModelClass(config).to(DEVICE)
    model = model.to(actual_dtype)   

    if gpu_id == 0:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if FORCE:
            OUTPUT_PATH = f"output/{MODEL}/{TRAINING_TYPE}/{n_params/1e6:.2f}M_force"
        else:
            OUTPUT_PATH = f"output/{MODEL}_{TRAINING_TYPE}_{n_params/1e6:.2f}M"

        OUTPUT_PATH = get_unique_filename(OUTPUT_PATH)

    logger = setup_logger(rank=gpu_id, log_dir=OUTPUT_PATH, filename=log_name)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER)

    if gpu_id == 0:

        logger.info("=" * 80)
        logger.info(f"Python: {platform.python_version()}, PyTorch: {torch.__version__}")
        logger.info(f"OS: {platform.system()} {platform.release()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
        logger.info(f"Model: {MODEL}")
        logger.info(f"Dtype: {actual_dtype}")
        logger.info(f"Config name: {config.config_name}")
        logger.info(f"Vocab size: {VOCAB_SIZE}")
        logger.info(f"Trainable params: {n_trainable/1e6:.2f}M / Total: {n_params/1e6:.2f}M")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Output path: {OUTPUT_PATH}")
        logger.info(f"Training type: {TRAINING_TYPE}")
        logger.info(f"Epochs: {EPOCH}, Batch size: {BATCH_SIZE}, LR: {LR}")
        logger.info(f"Eval every: {EVAL_STEPS} steps, Eval sample: {EVAL_SAMPLE}")
        logger.info(f"Num workers: {NUM_WORKERS}, Pin memory: {PIN_MEMORY}")
        logger.info(f"Shuffle: {SHUFFLE}, Drop last: {DROP_LAST}")
        logger.info(f"Training steps: {TRAINING_STEPS or 'auto (len(train_loader))'}")
        logger.info(f"Tokenizer: {HF_TOKENIZER}")
        logger.info(f"Checkpoint path: {checkpoint_path}")
        logger.info(f"Dataset source: {PRE_TRAINING_HF_DATA if TRAINING_TYPE == 'pre-train' else IT_HF_DATA}")

        logger.info(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)


    # Prepare training data for pre-train or instruction tuning
    if TRAINING_TYPE == "pre-train":
        train_loader, val_loader = get_pretrain_data(gpu_id, TOKEN_PATH, BATCH_SIZE, SHUFFLE,
                        DROP_LAST, NUM_WORKERS, PIN_MEMORY, SINGLE_FILE, logger)

    elif TRAINING_TYPE == "instruction-tuning":
        train_loader, val_loader = get_it_data(gpu_id, tokenizer, IT_HF_DATA, BATCH_SIZE, MAX_SEQ_LEN)

    else:
        print(f"Invalid TRAINING_TYPE argument: {TRAINING_TYPE}. Valid options are: pre-train, instruction-tuning.")

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
        num_epochs=EPOCH,
        training_steps=TRAINING_STEPS,
        eval_steps=EVAL_STEPS,
        eval_sample=EVAL_SAMPLE,
        learning_rate=LR,
        device=gpu_id,
        output_path=OUTPUT_PATH,
        force=FORCE,
    )

    # Start training
    try:
        trainer = Trainer(train_state,data_loaders,train_config, logger)
        trainer.train()

        if PUSH_TO_HUB:
            api = HfApi()
            api.upload_folder(
                folder_path=OUTPUT_PATH,          
                path_in_repo="logs",         
                repo_id=REPO_NAME,           
                repo_type="model",                
            )
            model.push_to_hub(REPO_NAME)
            tokenizer.push_to_hub(REPO_NAME)
            
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        raise
    finally:
        destroy_process_group()

if __name__ == "__main__":
    main()
