from tqdm import tqdm

import torch 
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class BenchmarkDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()

        dataset = load_dataset(data_path)
        self.tokenized_list = []
        for item in tqdm(dataset):
            text = item["text"]

            tokenized_text = tokenizer.encode(text)
            self.tokenized_list.append(torch.IntTensor(tokenized_text))


    def __len__(self):
        return len(self.tokenized_list)
    
    def __getitem__(self, index):
        return self.tokenized_list[index]
    

def get_benchmark_data(data_path, tokenizer, batch_size):

    bench_dataset = BenchmarkDataset(data_path=data_path, tokenizer=tokenizer)

    bench_loader = DataLoader(
        dataset=bench_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    return bench_loader