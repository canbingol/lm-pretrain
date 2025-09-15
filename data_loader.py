import os
import torch, gc
from itertools import islice
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset,get_dataset_config_info
from tqdm import tqdm

def prepare_train_data(hf_data_name,tokenizer,batch_size=5,context_len=512,len_data_row=50_000,val_ratio=0.06):
 
    dataset = load_dataset(hf_data_name, split="train", streaming=True)
    print("tokenizing...")

    train_tokens, val_tokens = [], []
    buf = []
    N_val = int(len_data_row * val_ratio)

    for i, ex in enumerate(tqdm(islice(dataset, len_data_row), total=len_data_row, desc=f"iterating {len_data_row}")):
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        ids = tokenizer.encode(text)
        buf.extend(ids)

        if i < N_val:
            val_tokens.extend(buf)
        else:
            train_tokens.extend(buf)
        buf.clear()

    print("tokenized")

    class TextDataset(Dataset):
        def __init__(self, tokens, max_length=8192, stride=8192):
            self.input_ids, self.target_ids = [], []
            for i in tqdm(range(0, len(tokens) - max_length, stride), desc="building chunks"):
                inp = tokens[i:i + max_length]
                tgt = tokens[i + 1:i + max_length + 1]
                self.input_ids.append(torch.tensor(inp))
                self.target_ids.append(torch.tensor(tgt))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]

    train_dataset = TextDataset(train_tokens, max_length=context_len, stride=context_len)
    val_dataset   = TextDataset(val_tokens,   max_length=context_len, stride=context_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    del dataset
    gc.collect()
    return train_loader,val_loader


def prepare_tokenizer_data(hf_data_name,text_column_name,output_path,len_data_row=100_000):
    filename = os.path.basename(hf_data_name)
    data_path = f"{output_path}/{filename}.txt"
    if os.path.exists(data_path):
        return data_path
 
    dataset = load_dataset(hf_data_name, split="train", streaming=True)
    print("preparing tokenizer data...")
    with open(data_path,"a") as f:
        for i, ex in enumerate(tqdm(islice(dataset, len_data_row), total=len_data_row, desc=f"iterating {len_data_row}")):
            text = (ex.get(text_column_name) or "").strip()
            if not text:
                continue
            if not text.endswith("."):            
                text = f"{text}."
            text = f"{text}\n"
            f.write(text)
    print(data_path)
    
    return data_path