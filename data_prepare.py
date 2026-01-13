import os
import torch, gc

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer

from utils import format_it_data

def create_tokens_file(hf_dataset: str ,hf_tokenizer, base_dir: str="./data/pretrain",test_split: int= 0.05,
                       tokens_chunks_size: int = 25_000_000, dtype: str = np.uint16, gpu_id: int= 0):
    
    tokenizer_name = hf_tokenizer.split("/")
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)

    save_dir = f"{base_dir}/{tokenizer_name}"
    os.makedirs(save_dir, exist_ok=True)
    mode = "train"
    data = load_dataset(hf_dataset, split="train")

    shard_id = 0
    token_list = []
    val_start = int(len(data) * (1 - test_split))

    bar = tqdm(data,desc="creating data shard") if gpu_id == 0 else data
    for i, item in enumerate(bar):

        tokens = tokenizer.encode(item["text"], add_special_tokens=False)
        token_list.extend(tokens)

        if i > val_start and mode =="train":
            if len(token_list) > 0:
                chunk = np.array(token_list, dtype=dtype)
                chunk.tofile(f"{base_dir}/{tokenizer_name}/{mode}_{shard_id:02d}.bin")
                shard_id += 1
                token_list = []

            mode = "validation"
            shard_id = 0

        if len(token_list) >= tokens_chunks_size or i == len(data)-1:
            chunk = np.array(token_list, dtype=np.uint16)
            chunk.tofile(f"{base_dir}/{tokenizer_name}/{mode}_{shard_id:02d}.bin")
            shard_id += 1
            token_list = []

    return save_dir

def prepare_pretrain_data(token_file_data_dir, batch_size, max_seq_len=512, pad_token=0, shuffle=False, drop_last=True, num_workers=0, pin_memory=True, single_file: int= None, gpu_id: int= 0, skip_ddp=False):

    class PretrainDataset(Dataset):
        def __init__(self, mode, exist_data_dir, max_seq_len, pad_token):
            super().__init__()

            # use only mode files
            data_files = [f for f in os.listdir(exist_data_dir) if f.startswith(mode)]

            if single_file is not None:
                data_files = [data_files[single_file]]

            self.input_ids, self.target_ids = [], []
            bar = tqdm(data_files, total= len(data_files), desc=f"iterating {mode} files") if gpu_id == 0 else data_files
            for file in bar:
                tokens = np.memmap(os.path.join(exist_data_dir, file), dtype=np.uint16, mode="r")

                for i in range(0,len(tokens) -1 ,max_seq_len):
                    input_ = tokens[i: i + max_seq_len]
                    target_ = tokens[i + 1: i + max_seq_len]
                    target_ = np.concatenate((target_,np.array([pad_token],dtype= np.uint16)))

                    if len(input_) < max_seq_len:
                        continue

                    self.input_ids.append(torch.tensor(input_))
                    self.target_ids.append(torch.tensor(target_))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            x = torch.as_tensor(self.input_ids[idx], dtype=torch.long)
            y = torch.as_tensor(self.target_ids[idx], dtype=torch.long)
            
            return x, y


    train_dataset = PretrainDataset(mode= "train", exist_data_dir= token_file_data_dir,
                                    max_seq_len= max_seq_len, pad_token= pad_token)

    val_dataset = PretrainDataset(mode= "validation", exist_data_dir= token_file_data_dir,
                                    max_seq_len= max_seq_len, pad_token= pad_token)

    if skip_ddp:
        train_dataloader = DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

        val_dataloader = DataLoader(
            dataset = val_dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = num_workers,
            pin_memory = pin_memory
        )
    else:
        train_dataloader = DataLoader(
            dataset = train_dataset,
            sampler = DistributedSampler(dataset=train_dataset, drop_last=True, shuffle=True),
            batch_size = batch_size,
            shuffle = False,
            drop_last = drop_last,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

        val_dataloader = DataLoader(
            dataset = val_dataset,
            sampler = DistributedSampler(dataset=val_dataset, drop_last=True, shuffle=True),
            batch_size = batch_size,
            shuffle = False,
            drop_last = drop_last,
            num_workers = num_workers,
            pin_memory = pin_memory
        )


    gc.collect()
    return train_dataloader, val_dataloader


def prepare_it_data(hf_dataset,tokenizer, batch_size, max_seq_len, pad_token, gpu_id, skipp_ddp:bool=False):

    def find_sub(seq, sub):
        sub = torch.as_tensor(sub)

        for i in range(len(seq) - len(sub) + 1):
            if seq[i:i+len(sub)].equal(sub):
                return i+1
        return 0

    class ITDataset(Dataset):
        def __init__(self, mode, data, tokenizer, max_seq_len, pad_token: int= 0, eos_token:int= 3, gpu_id=gpu_id):
            super().__init__()
            assistant_token_ids = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False)
            self.input_ids, self.target_ids = [], []
            bar = tqdm(data, total=len(data), desc=f"iterating IT data for {mode}") if gpu_id == 0 else data
            for item in bar:
                prompt = format_it_data(item["question"], item["input"], item["answer"])
                tokens = tokenizer.encode(prompt,  add_special_tokens=True)
                tokens += [pad_token] * (max_seq_len - len(tokens))
                tokens = tokens[:max_seq_len]

                input_ = tokens[:-1]
                target_ = tokens[1:]

                input_ = torch.tensor(input_)
                target_ = torch.tensor(target_)

                start_idx = find_sub(target_, assistant_token_ids) + 3
                mask_start = max(0, start_idx-1)
                target_[:mask_start] = -100
                self.input_ids.append(input_)
                self.target_ids.append(target_)


        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]

    data = load_dataset(hf_dataset, split= "train")
    val_ratio = 0.20
    val_len = int(len(data) * val_ratio)

    train_data = data.select(range(0, len(data) - val_len))
    val_data = data.select(range(len(data) - val_len, len(data)))

    train_dataset = ITDataset("train",train_data , tokenizer, max_seq_len, pad_token)

    val_dataset = ITDataset("validation", val_data, tokenizer, max_seq_len, pad_token)
    
    if skipp_ddp:
        train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = 0,
        pin_memory = True
        )

        val_dataloader = DataLoader(
            dataset = val_dataset,
            batch_size = batch_size,
            shuffle = False,
            drop_last = True,
            num_workers = 0,
            pin_memory = True
        )
    else:
        train_dataloader = DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            sampler = DistributedSampler(dataset=train_dataset, drop_last=True, shuffle=True),
            shuffle = False,
            drop_last = True,
            num_workers = 0,
            pin_memory = True
        )

        val_dataloader = DataLoader(
            dataset = val_dataset,
            batch_size = batch_size,
            sampler = DistributedSampler(dataset=val_dataset, drop_last=True, shuffle=True),
            shuffle = False,
            drop_last = True,
            num_workers = 0,
            pin_memory = True
        )

    gc.collect()
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer_path = "vngrs-ai/Kumru-2B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    pre_train_dataset = "canbingol/vngrs-web-corpus-500k"

    create_tokens_file(pre_train_dataset,"vngrs-ai/Kumru-2B", base_dir="./data/pretrain")
    exit()
    files = "./data/pretrain/default_tokenizer"
    train_loader, val_loader = prepare_pretrain_data(files,10)
    print(f"len train loader: {len(train_loader)}, len val loader: {len(val_loader)}")

    it_dataset = "merve/turkish_instructions"
    train_loader, val_loader = prepare_it_data(it_dataset,tokenizer, 10, 128, 0)
    print(f"len train loader: {len(train_loader)}, len val loader: {len(val_loader)}")
