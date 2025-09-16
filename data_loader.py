import os
import torch, gc
from itertools import islice
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

def prepare_train_data(hf_data_name,tokenizer,output_path,batch_size,context_len=512,len_data_row=1_000,val_ratio=0.06):
    sample_file = f"{output_path}/sample.txt"
    dataset = load_dataset(hf_data_name, split="train", streaming=True)
    dataset = dataset.with_format("torch")

    print("tokenizing...")

    train_tokens, val_tokens = [], []
    buf = []
    N_val = int(len_data_row * val_ratio)
    total_tokens = 0
    for i, ex in enumerate(tqdm(islice(dataset, len_data_row), total=len_data_row, desc=f"iterating {len_data_row}")):
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        text = text.lower()
        ids = tokenizer.encode(text)
        buf.extend(ids)
        total_tokens += len(ids)
        if i < N_val:
            val_tokens.extend(buf)
        else:
            train_tokens.extend(buf)
        buf.clear()
    print(f"{total_tokens} token tokenized")
    class TextDataset(Dataset):
        def __init__(self, tokens, max_length=512, stride=512,eos_token=3,bos_token=2):
            self.input_ids, self.target_ids = [], []
            for i in tqdm(range(0, len(tokens) - max_length, stride), desc="building chunks"):
                tokens = [bos_token] + tokens

                inp = tokens[i:i + max_length]
                tgt = tokens[i + 1:i + max_length] + [eos_token]
                self.input_ids.append(torch.tensor(inp))
                self.target_ids.append(torch.tensor(tgt))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]

    train_dataset = TextDataset(train_tokens, max_length=context_len, stride=context_len)
    val_dataset   = TextDataset(val_tokens,   max_length=context_len, stride=context_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    with open(sample_file, "w", encoding="utf-8") as f:
        for i in range(2):
            inp, tgt = train_dataset[i]

            inp_ids = inp.tolist()
            tgt_ids = tgt.tolist()

            inp_text = tokenizer.decode(inp_ids)
            tgt_text = tokenizer.decode(tgt_ids)

            f.write(f"--- Sample {i+1} ---\n")
            f.write(f"Input IDs:  {inp_ids}\n")
            f.write(f"Target IDs: {tgt_ids}\n")
            f.write(f"Input Text:  {inp_text}\n")
            f.write(f"Target Text: {tgt_text}\n\n")

    del dataset
    gc.collect()
    return train_loader,val_loader

def prepare_tokenizer_data(hf_data_name,text_column_name,output_path,len_data_row=1_500_000):
    filename = os.path.basename(hf_data_name)
    data_path = f"{output_path}/{filename}.txt"
    if os.path.exists(data_path):
        return data_path
 
    dataset = load_dataset(hf_data_name, split="train", streaming=True)
    print("preparing tokenizer data...")
    with open(data_path,"a") as f:
        for i, ex in enumerate(tqdm(islice(dataset, len_data_row), total=len_data_row, desc=f"iterating {len_data_row}")):
            text = (ex.get(text_column_name) or "").strip()
            text = text.lower()

            if not text:
                continue
            if not text.endswith("."):            
                text = f"{text}."
            text = f"{text}\n"
            f.write(text)
    print(data_path)
    
    return data_path