import sys
import os

import torch
import torch.nn.functional as f
import argparse


from models.qwen3 import Qwen3,Qwen3Config
from inference import generate
from train_tokenizer import TrainTokenizer
from data_loader import prepare_train_data, prepare_tokenizer_data
from pre_train import lm_pretrain
import sentencepiece as spm

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

    parser.add_argument('--max-train-step', type=int, default=None,
                        help='Maximum number of training steps (if None, computed automatically)')
    parser.add_argument('--max-eval-sample', type=int, default=2_000,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--max-eval-step', type=int, default=200,
                        help='Number of evaluation steps per evaluation run')
    parser.add_argument('--vocab-size', type=int, default=10_000,
                        help='Vocabulary size of the model/tokenizer')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training/evaluation')

    parser.add_argument('--save-step', type=int, default=5_000,
                        help='Save model checkpoint every N steps')
    parser.add_argument('--logging', action="store_true", default=False,
                        help='Enable logging during training')
    parser.add_argument('--cpt', type=str, default=None,
                        help='Path to checkpoint for continuing pre-training')
    parser.add_argument('--text-column_name', type=str, default="text",
                        help='Name of the text column in the HF dataset')

    parser.add_argument('--model-info', action="store_true", default=False,
                        help='Print model configuration information')
    parser.add_argument('--train-tokenizer', action="store_true", default=False,
                        help='Enable training a new tokenizer')
    parser.add_argument('--inference', action="store_true", default=False,
                        help='Run inference instead of training')

    parser.add_argument('--hf-data', type=str, default=None,
                        help='Name or path of the Hugging Face dataset')

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    model_map = {
        "qwen3": (Qwen3Config, Qwen3),
    }
    ConfigClass, ModelClass = model_map[args.model]
    config = ConfigClass()

    DEVICE = config.device

    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    config.vocab_size = args.vocab_size
    config.batch_size = args.batch_size

    print(f"config name : {config.config_name}\nconfig vocab_size : {config.vocab_size}")

    print("device : ",DEVICE)
    SAVE_STEP = args.save_step
    model = ModelClass(config).to(DEVICE)
    EPOCH = args.epoch
    MAX_TRAIN_STEP = args.max_train_step
    MAX_EVAL_STEP = args.max_eval_step
    LR = args.lr
    LOGGING = args.logging
    MODEL_INFO =  args.model_info

    OUTPUT_PATH = f"{args.model}"
    criterion = f.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    tokenizer_data_path = prepare_tokenizer_data(args.hf_data,args.text_column_name,OUTPUT_PATH)
    tokenizer_path = f"{OUTPUT_PATH}/{OUTPUT_PATH}.tokenizer.model"
    if args.train_tokenizer or not os.path.exists(tokenizer_path):
        tokenizer_path = TrainTokenizer(config,tokenizer_data_path,OUTPUT_PATH).get_model_path()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    assert tokenizer_path is not None, f"there is no tokenizer.model in {tokenizer_path}, if u want to create a tokenizer use --train-tokenizer"
    print("args.inference : ",args.inference)
    if args.inference:

        response = generate(model, tokenizer, args.prompt, device="cuda", max_new_tokens=64)
        print(f"Model Response :\n{response}")
        exit()

    train_loader, val_loader = prepare_train_data(args.hf_data,tokenizer,OUTPUT_PATH, batch_size=config.batch_size,context_len=config.max_seq_len)

    if MAX_TRAIN_STEP is not None:
        MAX_TRAIN_STEP = min(len(train_loader),MAX_TRAIN_STEP)
    else:
        MAX_TRAIN_STEP = len(train_loader)
        

    lm_pretrain(model,train_loader,val_loader,criterion,optimizer,EPOCH,MAX_TRAIN_STEP,MAX_EVAL_STEP,LOGGING,DEVICE,SAVE_STEP,OUTPUT_PATH,MODEL_INFO)

if __name__ == "__main__":

    main()