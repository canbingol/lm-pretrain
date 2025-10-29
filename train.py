import time, os, sys
from dataclasses import astuple
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import (
    TrainState,
    TrainConfig,
    DataLoaders,
    clear_gpu_memory,
    logging
)

def calcc(model, input_batch, target_batch, device):
    # Cast & move to device
    input_batch = input_batch.long().to(device, non_blocking=True)
    target_batch = target_batch.long().to(device, non_blocking=True)

    # Forward pass
    logits = model(input_batch)
    del input_batch

    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_batch.reshape(-1),
        ignore_index=0
    )

    del target_batch, logits
    return loss


def calcc_loss_batch(model, input_batch, target_batch, device):

    loss = calcc(model,input_batch,target_batch,device)
    return loss

def calc_loader_loss(model, data_loader, device, val_loss_file, num_batch=None, type_train=True):
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

            if not type_train:
                progress = (i + 1) / num_batch
                sys.stdout.write(
                    f"\r[Eval {i+1:4d}/{num_batch}]  "
                    f"Progress: {progress*100:6.2f}%  "
                    f"Step Loss: {loss}  "
                )
                sys.stdout.flush()
                with open(val_loss_file,"a")as f:
                    f.write(f"{loss}\n")
            total_loss += loss.item()
            del loss
        else:
            break
        clear_gpu_memory()
    return total_loss / num_batch

def evaluate_model(model, val_loader, device, eval_iter, val_loss_file):
    model.eval()
    with torch.no_grad():

        val_loss = calc_loader_loss(model, val_loader, device, val_loss_file, num_batch=eval_iter, type_train=False)

    model.train()
    return val_loss


def train_model(train_state: TrainState,data_loaders: DataLoaders, training_config: TrainConfig, logger):
    start_time = time.time()
    model,checkpoint_path  = astuple(train_state)
    train_loader,val_loader = astuple(data_loaders)
    num_epochs,training_steps,eval_steps,eval_sample,learning_rate,device,output_path,force = astuple(training_config)

    train_loss_file = f"{output_path}/train_loss.txt"
    val_loss_file = f"{output_path}/validation_loss.txt"

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
        logger.info(f"GPU {gpu_id}: Resuming training from snapshot.")

    train_losses, val_losses, track_tokens_seen = [],[],[]
    token_seen , global_step = 0, -1

    best_val_loss = float("inf")
    optimizer = torch.optim.AdamW(model.parameters(),1e-4, weight_decay=0.1)

    for epoch in range(num_epochs):
        model.train()
        torch.autograd.set_detect_anomaly(True)
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if gpu_id == 0 else train_loader
        for input_batch, target_batch in pbar:
            torch.autograd.set_detect_anomaly(True)
            loss = calcc_loss_batch(model,input_batch,target_batch,device)

            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1
            token_seen += input_batch.numel()

            if gpu_id == 0:
                pbar.set_postfix({"train loss":f"{loss.item():.4f}"})
                with open(train_loss_file,"a")as f:
                    f.write(f"{global_step}, {loss.item()}\n")

            if  (global_step > 0 and global_step % eval_steps == 0):
                if gpu_id == 0:
                    print("Calculating loss...", end="", flush=True)

                    val_loss = evaluate_model(
                        model,val_loader,device,eval_sample, val_loss_file
                    )
                    val_losses.append(val_loss)

                    track_tokens_seen.append(token_seen)
                    sys.stdout.write("\r")
                    sys.stdout.write(" " * 50 + "\r")
                    sys.stdout.flush()
                    logger.info(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {loss:.3f} Val loss {val_loss:.3f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                    "epoch": epoch,
                    "step":global_step,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": loss,
                    "val_loss": val_loss
                },f"{output_path}/{model.module.name}_best_model.pt")

        clear_gpu_memory()

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    if gpu_id == 0:
        logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")
    return train_losses, val_losses, track_tokens_seen

def trainer(train_state: TrainState,data_loaders: DataLoaders, training_config: TrainConfig, logger):
    train_losses, val_losses, tokens_seen = train_model(
        train_state,data_loaders,training_config, logger
    )
    return train_losses,val_losses,tokens_seen
