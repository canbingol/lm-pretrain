# stdlib
import os
import platform
from datetime import datetime

# third-party
import torch
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

# local modules - top-level logic
from inference import generate
from train.trainer import Trainer
from data_prepare import (
    prepare_pretrain_data,
    create_tokens_file,
    prepare_it_data,
)
from utils import (
    load_model,
    build_argparser,
    TrainState,
    TrainConfig,
    DataLoaders,
    merge_args_with_yaml,
    load_yaml,
    setup_logger,
    get_unique_filename
)

# model registry - separated by domain
from models.qwen3 import Qwen3, Qwen3Config
from models.llama2 import LLaMA2, LlamaConfig
from models.gpt2 import GPTConfig, GPTModel


# Seed info
torch.manual_seed(42)


def ddp_setup(WORLD_SIZE):
    
    backend = "gloo" if WORLD_SIZE < 2 else "nccl"

    init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_pretrain_data(gpu_id, saved_tokens_path, BATCH_SIZE, SHUFFLE,
                        DROP_LAST, NUM_WORKERS, PIN_MEMORY, SINGLE_FILE, logger):
    """
    This Function create train and validation loaders for pre training
    """
    train_loader, val_loader = prepare_pretrain_data(
        token_file_data_dir=saved_tokens_path, batch_size=BATCH_SIZE,shuffle=SHUFFLE,
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

    parser = build_argparser()
    args = parser.parse_args()
    if args.config:
        yaml_cfg = load_yaml(args.config)
        args = merge_args_with_yaml(args, yaml_cfg)

    WORLD_SIZE = int(args.world_size)
    TRAINING_TYPE = args.training_type
    MODEL = str(args.model)
    VOCAB_SIZE = int(args.vocab_size)
    BATCH_SIZE = int(args.batch_size)
    EPOCH = int(args.epoch)
    EVAL_SAMPLE = int(args.eval_sample)
    EVAL_STEPS = int(args.eval_steps)
    LR = float(args.lr)
    SHUFFLE = bool(args.shuffle)
    DROP_LAST = bool(args.drop_last)
    NUM_WORKERS = int(args.num_workers)
    PIN_MEMORY = bool(args.pin_memory)
    SINGLE_FILE = str(args.single_file) if args.single_file is not None else None
    SAVED_TOKEN_PATH = str(args.saved_token_path)
    PRE_TRAINING_HF_DATA = str(args.pre_training_hf_data)
    IT_HF_DATA = str(args.it_hf_data)
    HF_TOKENIZER = str(args.hf_tokenizer)
    FORCE = bool(args.force)
    INFERENCE = bool(args.inference)
    PROMPT = str(args.prompt)
    checkpoint_path = str(args.checkpoint)
    MAX_NEW_TOKENS = int(args.max_new_tokens)
    MAX_SEQ_LEN = int(args.max_seq_len)
    TRAINING_STEPS = (
    int(args.training_steps)
    if args.training_steps not in [None, "None", "none", ""]
    else None
    )
    ddp_setup(WORLD_SIZE)

    saved_tokens_path = SAVED_TOKEN_PATH

    # If given saved token path not exist create .bin files with given hf dataset
    if not os.path.exists(saved_tokens_path) or len(os.listdir(saved_tokens_path)) == 0:
        saved_tokens_path = create_tokens_file(PRE_TRAINING_HF_DATA, HF_TOKENIZER)


    model_map = {
        "qwen3": (Qwen3Config, Qwen3),
        "llama2":(LlamaConfig,LLaMA2),
        "gpt2": (GPTConfig, GPTModel)

    }

    # Initialize model with cuda device
    ConfigClass, ModelClass = model_map[MODEL]
    config = ConfigClass()

    config.vocab_size = VOCAB_SIZE
    config.batch_size = BATCH_SIZE

    gpu_id = int(os.environ["LOCAL_RANK"])
    DEVICE = torch.device(f"cuda:{gpu_id}")
    config.device = DEVICE
    log_name = f"train_{MODEL}_{TRAINING_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = setup_logger(rank=gpu_id, filename=log_name)

    model = ModelClass(config).to(DEVICE)

    if gpu_id == 0:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if FORCE:
            OUTPUT_PATH = f"output/{MODEL}/{TRAINING_TYPE}/{n_params/1e6:.2f}M_force"
        else:
            OUTPUT_PATH = f"output/{MODEL}_{TRAINING_TYPE}_{n_params/1e6:.2f}M"

        OUTPUT_PATH = get_unique_filename(OUTPUT_PATH)

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


    # Make inference
    if INFERENCE:
        if gpu_id == 0:
            model.module = load_model(model,inference=True,checkpoint_path=checkpoint_path,device=DEVICE)
            logger.info("Generation Response...")
            response = generate(model, tokenizer,PROMPT , device=DEVICE, max_new_tokens=MAX_NEW_TOKENS)
            logger.info(f"Model Response :\n{response}")
            exit()

    # Prepare training data for pre-train or instruction tuning
    if TRAINING_TYPE == "pre-train":
        train_loader, val_loader = get_pretrain_data(gpu_id, saved_tokens_path, BATCH_SIZE, SHUFFLE,
                        DROP_LAST, NUM_WORKERS, PIN_MEMORY, SINGLE_FILE, logger)

    else:
        train_loader, val_loader = get_it_data(gpu_id, tokenizer, IT_HF_DATA, BATCH_SIZE, MAX_SEQ_LEN)

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

    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        raise
    finally:
        destroy_process_group()

if __name__ == "__main__":
    main()
