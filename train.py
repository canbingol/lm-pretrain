import time, os, sys
from dataclasses import astuple
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from utils import (
    TrainState,
    TrainConfig,
    DataLoaders,
    clear_gpu_memory,
)

def calcc(model,input_batch,target_batch,device):

    input_batch = input_batch.to(device, non_blocking=True)
    input_batch = input_batch.to(torch.long)
    logits = model(input_batch)
    del input_batch
    target_batch = target_batch.to(device, non_blocking=True)

    loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_batch.reshape(-1).long()
        )
    del target_batch,logits
    return loss

def calcc_loss_batch(model, input_batch, target_batch, device):

    loss = calcc(model,input_batch,target_batch,device)
    return loss

def calc_loader_loss(model, data_loader, device, num_batch=None):
    total_loss = 0

    if len(data_loader) == 0:
        print("in calc_loader_loss len(data_loader)=0")
        return float("nan")
    elif num_batch is None:
        num_batch = len(data_loader)
    else:
        num_batch = min(num_batch, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batch:
            loss = calcc_loss_batch(model,input_batch,target_batch,device)
            total_loss += loss.item()
            del loss
        else:
            break
        clear_gpu_memory()
    return total_loss / num_batch

def evaluate_model(model,train_loader, val_loader,device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loader_loss(model,train_loader,device,num_batch=eval_iter)
        val_loss = calc_loader_loss(model,val_loader,device,num_batch=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(train_state: TrainState,data_loaders: DataLoaders, training_config: TrainConfig):
    start_time = time.time()
    model,checkpoint_path  = astuple(train_state)
    train_loader,val_loader = astuple(data_loaders)
    num_epochs,training_steps,eval_steps,eval_sample,learning_rate,device,output_path,force = astuple(training_config)

    gpu_id = device
    model = DDP(model,device_ids=[gpu_id])
    del train_state, data_loaders, training_config

    warmup_steps = 100
    min_lr = 3e-5

    optimizer = torch.optim.AdamW(model.parameters(),learning_rate, weight_decay=0.1)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=training_steps - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

    if os.path.exists(checkpoint_path) and not force and gpu_id == 0:
        loc = f"cuda:{gpu_id}"
        snapshot = torch.load(checkpoint_path, map_location=loc)
        model.module.load_state_dict(snapshot["model_state_dict"])
        optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        scheduler.load_state_dict(snapshot["scheduler_state_dict"])
        print(f"GPU {gpu_id}: Resuming training from snapshot.")

    train_losses, val_losses, track_tokens_seen = [],[],[]
    token_seen , global_step = 0, -1

    best_val_loss = float("inf")
    optimizer = torch.optim.AdamW(model.parameters(),1e-4, weight_decay=0.1)

    for epoch in range(num_epochs):
        model.train()
        torch.autograd.set_detect_anomaly(True)
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader) if gpu_id == 0 else train_loader
        for input_batch, target_batch in pbar:
            loss = calcc_loss_batch(model,input_batch,target_batch,device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1
            token_seen += input_batch.numel()

            if  global_step == 0 or (global_step > 0 and (global_step % eval_steps == 0 or global_step % training_steps == 0)):
                if gpu_id == 0:
                    print("Calculating loss...", end="", flush=True)

                    train_loss, val_loss = evaluate_model(
                        model,train_loader,val_loader,device,eval_sample
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(token_seen)
                    sys.stdout.write("\r")
                    sys.stdout.write(" " * 50 + "\r")
                    sys.stdout.flush()
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                    "epoch": epoch,
                    "step":global_step,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss
                },f"{output_path}/{model.module.name}_best_model.pt")

            dist.barrier()
        clear_gpu_memory()

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    if gpu_id == 0:
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    return train_losses, val_losses, track_tokens_seen

def trainer(train_state: TrainState,data_loaders: DataLoaders, training_config: TrainConfig):
    train_losses, val_losses, tokens_seen = train_model(
        train_state,data_loaders,training_config
    )
    return train_losses,val_losses,tokens_seen
