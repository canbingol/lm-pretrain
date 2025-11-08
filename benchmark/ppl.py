from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from models.qwen3 import Qwen3,Qwen3Config
from models.llama2 import LLaMA2, LlamaConfig
from models.gpt2 import GPTConfig, GPTModel

from data_prepare import prepare_pretrain_data

class PPLBenchmark:

    def __init__(self, model_path, model_type, tokenizer_name, device, batch_size, data_path="./data/benchmark/ppl"):
        self.batch_size = batch_size
        self.data_path = data_path
        self.device = device

        model_map = {
        "qwen3": (Qwen3Config, Qwen3),
        "llama2":(LlamaConfig,LLaMA2),
        "gpt2": (GPTConfig, GPTModel)

    }
        
        ConfigClass, ModelClass = model_map[model_type]
        config = ConfigClass()

        self.model = ModelClass(config).to(self.device).eval()

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def benchmark(self):
        _, data_loader = prepare_pretrain_data(token_file_data_dir="./data/pretrain/default_tokenizer", batch_size=self.batch_size, skip_ddp=True) 

        ppl_list = []
        acc_list = []
        loss_list = []
        entropy_list = []
        distinct_list = []

        with torch.no_grad():

            for input_batch, target_batch in tqdm(data_loader):

                input_batch = input_batch.long().to(self.device, non_blocking=True)
                target_batch = target_batch.long().to(self.device, non_blocking=True)

                logits = self.model(input_batch)

                loss = torch.nn.functional.cross_entropy(input=logits.reshape(-1, logits.shape[-1]),
                                                         target= target_batch.reshape(-1))
                
                ppl = torch.exp(loss)
                preds = logits.argmax(dim=-1)
                acc = (preds == target_batch).float().mean()
                tokens = preds.flatten().tolist()

                distinct_ratio = len(set(tokens)) / len(tokens)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()

                loss_list.append(loss)                
                ppl_list.append(ppl)
                acc_list.append(acc)
                distinct_list.append(distinct_ratio)
                entropy_list.append(entropy)

                log = f"{loss:.4f},{ppl:.4f},{acc:.4f},{distinct_ratio:.4f},{entropy:.4f}"

                with open("log.txt", "a", encoding="utf-8") as f:
                    f.write(f"{log}\n")
                    
            print("="*30)
            print("Average Result")
            print(f"Loss : {sum(loss_list) / len(loss_list):.4f}")
            print(f"Perplexity : {sum(ppl_list) / len(ppl_list):.4f}")
            print(f"Accuracy : {sum(acc_list) / len(acc_list):.4f}")
            print(f"Distinct Ratio : {sum(distinct_list) / len(distinct_list):.4f}")
            print(f"Entropy : {sum(entropy_list) / len(entropy_list):.4f}")
            print("="*30)

if __name__ == "__main__":
    MODEL_PATH = "./output/gpt2_pre-train/GPT2_best_model_data500k_loss4.3.pt"
    MODEL_TYPE = "gpt2"
    TOKENIZER_NAME = "vngrs-ai/Kumru-2B"
    DEVICE = "cuda"
    BATCH_SIZE = 10

    ppl = PPLBenchmark(model_path=MODEL_PATH, tokenizer_name=TOKENIZER_NAME, model_type=MODEL_TYPE,
                       device=DEVICE, batch_size=BATCH_SIZE)
    
    ppl.benchmark()

