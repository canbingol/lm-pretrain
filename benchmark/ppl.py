from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from utils import get_benchmark_data

from models.deepseekV2 import Deepseek, DeepseekConfig
from models.qwen3 import Qwen3,Qwen3Config
from models.gemma2 import Gemma2, GemmaConfig
from models.llama2 import LLaMA2, LlamaConfig
from models.gpt2 import GPTConfig, GPTModel


class PPLBenchmark:

    def __init__(self, model_path, model_type, tokenizer_name, device, batch_size, data_path="./data/benchmark/ppl"):
        self.batch_size = batch_size
        self.data_path = data_path

        model_map = {
        "qwen3": (Qwen3Config, Qwen3),
        "deepseek":(DeepseekConfig, Deepseek),
        "gemma2":(GemmaConfig,Gemma2),
        "llama2":(LlamaConfig,LLaMA2),
        "gpt2": (GPTConfig, GPTModel)

    }
        
        ConfigClass, ModelClass = model_map[model_type]
        config = ConfigClass()

        self.model = ModelClass(config).to(device).eval()

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def benchmark(self):
        data_loader = get_benchmark_data(data_path=self.data_path, tokenizer=self.tokenizer, batch_size=self.batch_size)    
        
        with torch.no_grad():

            for input_batch, target_batch in tqdm(data_loader):
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
                
                print(f"loss: {loss}, perplexity: {ppl}, accuracy : {acc}, distinct_ratio : {distinct_ratio}, entropy : {entropy}")




