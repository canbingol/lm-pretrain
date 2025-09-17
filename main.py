import os

import torch
import torch.nn.functional as F

from models.deepseekV2 import Deepseek, DeepseekConfig
from models.qwen3 import Qwen3,Qwen3Config
from models.gemma2 import Gemma2, GemmaConfig
from models.llama2 import LLaMA2, LlamaConfig

from inference import generate
from data_loader import prepare_train_data
from pre_train import trainer
from utils import (
    get_tokenizer, 
    load_model, model_size_info,
    build_argparser,
    TrainState,
    TrainConfig,
    DataLoaders
)
from warnings import filterwarnings
filterwarnings("ignore")
def main():
    parser = build_argparser()
    args = parser.parse_args()
    model_map = {
        "qwen3": (Qwen3Config, Qwen3),
        "deepseek":(DeepseekConfig, Deepseek),
        "gemma":(GemmaConfig,Gemma2),
        "llama2":(LlamaConfig,LLaMA2)

    }
    ConfigClass, ModelClass = model_map[args.model]
    config = ConfigClass()

    DEVICE = config.device

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise Exception("Cuda is not available")
    
    config.vocab_size = args.vocab_size
    config.batch_size = args.batch_size

    EPOCH = args.epoch
    TRAINING_STEPS = args.training_steps
    EVAL_SAMPLE  =args.eval_sample
    EVAL_STEPS = args.eval_steps
    LR = args.lr
 
    FORCE = args.force

    if FORCE:
        OUTPUT_PATH = f"output/{args.model}_force"
    else:
        OUTPUT_PATH = f"output/{args.model}"

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    model = ModelClass(config).to(DEVICE)

    print(f"config name : {config.config_name}\nconfig vocab_size : {config.vocab_size}")
    print("device : ",DEVICE)
    tokenizer = get_tokenizer(args.hf_data,args.text_column_name,OUTPUT_PATH,config.vocab_size,args.train_tokenizer)

    checkpoint_path = f"{OUTPUT_PATH}/{args.model}_best_model.pt"

    if args.model_info:
        model_size_info(model)

    if args.inference:
        model = load_model(model,inference=True,checkpoint_path=checkpoint_path,device=DEVICE)
        print("Generation Response...")
        response = generate(model, tokenizer, args.prompt, device=DEVICE, max_new_tokens=args.max_new_tokens)
        print(f"Model Response :\n{response}")
        exit()

    train_loader, val_loader = prepare_train_data(args.hf_data,tokenizer,OUTPUT_PATH, batch_size=config.batch_size,context_len=config.max_position_embeddings)
    
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
        device=DEVICE,
        output_path=OUTPUT_PATH,
        force=FORCE,
    )

    trainer(train_state,data_loaders,train_config)
if __name__ == "__main__":

    main()