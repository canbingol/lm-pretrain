import os

import torch
import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from transformers import AutoTokenizer

from models.deepseekV2 import Deepseek, DeepseekConfig
from models.qwen3 import Qwen3,Qwen3Config
from models.gemma2 import Gemma2, GemmaConfig
from models.llama2 import LLaMA2, LlamaConfig

from inference import generate
from data_prepare import prepare_pretrain_data, create_tokens_file, prepare_it_data
from train import trainer
from utils import (
    load_model, model_size_info,
    build_argparser,
    TrainState,
    TrainConfig,
    DataLoaders,
    merge_args_with_yaml,
    load_yaml
)
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

from warnings import filterwarnings
filterwarnings("ignore")
def main():
    ddp_setup()

    parser = build_argparser()
    args = parser.parse_args()
    if args.config:
        yaml_cfg = load_yaml(args.config)
        args = merge_args_with_yaml(args, yaml_cfg)

    TRAINING_TYPE = args.training_type
    MODEL = str(args.model)
    VOCAB_SIZE = int(args.vocab_size)
    BATCH_SIZE = int(args.batch_size)
    EPOCH = int(args.epoch)
    EVAL_SAMPLE = int(args.eval_sample)
    EVAL_STEPS = int(args.eval_steps)
    LR = float(args.lr)
    MODEL_INFO = bool(args.model_info)
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
    MAX_NEW_TOKENS = int(args.max_new_tokens)
    MAX_SEQ_LEN = int(args.max_seq_len)
    TRAINING_STEPS = (
    int(args.training_steps)
    if args.training_steps not in [None, "None", "none", ""]
    else None
)

    model_map = {
        "qwen3": (Qwen3Config, Qwen3),
        "deepseek":(DeepseekConfig, Deepseek),
        "gemma2":(GemmaConfig,Gemma2),
        "llama2":(LlamaConfig,LLaMA2)

    }
    saved_tokens_path = SAVED_TOKEN_PATH

    if not os.path.exists(saved_tokens_path) or len(os.listdir(saved_tokens_path)) == 0:
        saved_tokens_path = create_tokens_file(PRE_TRAINING_HF_DATA, HF_TOKENIZER)

    ConfigClass, ModelClass = model_map[MODEL]
    config = ConfigClass()

    gpu_id = int(os.environ["LOCAL_RANK"])
    config.device = gpu_id
    DEVICE = config.device

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise Exception("Cuda is not available")

    config.vocab_size = VOCAB_SIZE
    config.batch_size = BATCH_SIZE


    if FORCE:
        OUTPUT_PATH = f"output/{MODEL}_force"
    else:
        OUTPUT_PATH = f"output/{MODEL}"

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    model = ModelClass(config).to(DEVICE)
    if gpu_id == 0:
        print(f"config name : {config.config_name}\nconfig vocab_size : {config.vocab_size}")
        print("device : ",DEVICE)
    #tokenizer = get_tokenizer(args.hf_data,args.text_column_name,OUTPUT_PATH,config.vocab_size,args.train_tokenizer,gpu_id)
    tokenizer = AutoTokenizer.from_pretrained("./thirdpart/kumru_tokenizer")
    checkpoint_path = f"{OUTPUT_PATH}/{MODEL}_best_model.pt"

    if MODEL_INFO:
        if gpu_id == 0:
            model_size_info(model)

    if INFERENCE:
        if gpu_id == 0:
            model.module = load_model(model,inference=True,checkpoint_path=checkpoint_path,device=DEVICE)
            print("Generation Response...")
            response = generate(model, tokenizer,PROMPT , device=DEVICE, max_new_tokens=MAX_NEW_TOKENS)
            print(f"Model Response :\n{response}")
            exit()
    if TRAINING_TYPE == "pre-train":
        train_loader, val_loader = prepare_pretrain_data(
            token_file_data_dir=saved_tokens_path, batch_size=BATCH_SIZE,shuffle=SHUFFLE,
            drop_last=DROP_LAST, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, single_file=SINGLE_FILE,
            gpu_id=gpu_id
            )

    else:
        train_loader, val_loader = prepare_it_data(
            hf_dataset=IT_HF_DATA, tokenizer=tokenizer, batch_size=BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN, pad_token=tokenizer.pad_token_id, gpu_id=gpu_id
        )

    TRAINING_STEPS = len(train_loader) if not TRAINING_STEPS else TRAINING_STEPS
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

    trainer(train_state,data_loaders,train_config)
    destroy_process_group()
if __name__ == "__main__":
    main()
