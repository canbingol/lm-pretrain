"""
TODO: Refactor this file to implement generation using model.generate.
"""


import argparse, sys
import torch
from transformers import AutoTokenizer

from models.decoder_model import DecoderCausalLM, ModelConfig


@torch.no_grad()
def generate(model, tokenizer, prompt, device="cuda", max_new_tokens=64,temprature=0.67,top_k=100 ):
    model.to(device).eval()
    input_ids = tokenizer.encode(prompt,return_tensors="pt")
    input_ids = input_ids[None,:].to(device) if input_ids.ndim == 1 else input_ids.to(device)

    if model.config.attn_type == "flash_attn":
        model = model.to(torch.bfloat16)

    for i in range(max_new_tokens):
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
        sys.stdout.write(f"\r[Eval {i+1:4d}/{max_new_tokens}] {i+1}th token generated...")
        sys.stdout.flush()
    message = tokenizer.decode(input_ids.squeeze(0))
    print(message)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument('--model', type=str, choices=["qwen3","llama2","gpt2"])
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--tokenizer-path', type=str)
    parser.add_argument('--prompt', type=str)
    args = parser.parse_args()

    args = "TEMP"

    cp = args.checkpoint
    tokenizer_path = args.tokenizer_path
    prompt = args.prompt

    model_map = {
        "decoder": (ModelConfig, DecoderCausalLM)

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
