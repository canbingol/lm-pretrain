
import argparse
import torch
from transformers import AutoTokenizer

from utils import (
    load_yaml,
    merge_args_with_yaml
)
from models.deepseekV2 import Deepseek, DeepseekConfig
from models.qwen3 import Qwen3,Qwen3Config
from models.gemma2 import Gemma2, GemmaConfig
from models.llama2 import LLaMA2, LlamaConfig

@torch.no_grad()
def generate(model, tokenizer, prompt, device="cpu", max_new_tokens=64,temprature=1.0,top_k=50 ):
    model.to(device).eval()
    input_ids = tokenizer.encode(prompt,return_tensors="pt")
    input_ids = input_ids[None,:].to(device) if input_ids.ndim == 1 else input_ids.to(device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        logits = logits[:,-1,:] / temprature

        if top_k is not None:
            topk_vals, topk_ids = torch.topk(logits,top_k)
            mask = torch.full_like(logits,float("-inf"))
            mask.scatter_(1,topk_ids,topk_vals)
            logits = mask

        probs = torch.nn.functional.softmax(logits, dim=-1)
        new_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids,new_token),dim=-1)
    message = tokenizer.decode(input_ids.squeeze(0))
    print(message)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument('--model', type=str, choices=["qwen3","deepseek","gemma2","llama2"])
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--tokenizer-path', type=str)
    parser.add_argument('--prompt', type=str)
    args = parser.parse_args()

    yaml_cfg = load_yaml(args.config)
    args = merge_args_with_yaml(args,yaml_cfg)

    cp = args.checkpoint
    tokenizer_path = args.tokenizer_path
    prompt = args.prompt

    model_map = {
        "qwen3": (Qwen3Config, Qwen3),
        "deepseek":(DeepseekConfig, Deepseek),
        "gemma2":(GemmaConfig,Gemma2),
        "llama2":(LlamaConfig,LLaMA2)

    }
    snapshot = torch.load(cp,map_location="cuda")
    ConfigClass, ModelClass = model_map[args.model]
    config = ConfigClass()
    model = ModelClass(config).to("cuda")
    model.load_state_dict(snapshot["model_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    generate(model,tokenizer,prompt,temprature=0.6)

if __name__ == "__main__":
    main()
