import time, os
from itertools import islice
from dataclasses import astuple
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tqdm import tqdm

from utils import (
    load_model, 
    TrainState,
    TrainConfig,
    DataLoaders,
    clear_gpu_memory
)

def calcc(model,input_batch,target_batch,device):
    total_loss = 0
    len_batch = len(input_batch)

    input_batch = input_batch.to(device, non_blocking=True)
    logits = model(input_batch)
    del input_batch
    target_batch = target_batch.to(device, non_blocking=True)

    logits = logits.reshape(logits.shape[0] * logits.shape[1],-1)
    target_batch = target_batch.reshape(target_batch.shape[0] * target_batch.shape[1])

    loss = torch.nn.functional.cross_entropy(logits,target_batch)
    total_loss += loss
    del target_batch,logits,loss

    return total_loss / len_batch


def calcc_loss_batch(model, input_batch, target_batch, device):
    loss = calcc(model,input_batch,target_batch,device)
    return loss


def calc_loader_loss(model, data_loader, device, num_batch=None):
    total_loss = 0

    if len(data_loader) == 0:
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

    model, optimizer, scheduler,_  = astuple(train_state)
    train_loader,val_loader = astuple(data_loaders)
    num_epochs,training_steps,eval_steps,eval_sample,_,device,output_path,_ = astuple(training_config)

    del train_state, data_loaders, training_config
    
    train_losses, val_losses, track_tokens_seen = [],[],[]
    token_seen , global_step = 0, -1

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in tqdm(islice(train_loader,training_steps), total=training_steps):

            loss = calcc_loss_batch(model,input_batch,target_batch,device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            token_seen += input_batch.numel()

            if global_step > 0 and (global_step % eval_steps == 0 or global_step % training_steps == 0):

                train_loss, val_loss = evaluate_model(
                    model,train_loader,val_loader,device,eval_sample
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                "epoch": epoch,
                "step":global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,  
                "val_loss": val_loss
            },f"{output_path}/{model.name}_best_model.pt")
                    
        clear_gpu_memory()
    return train_losses, val_losses, track_tokens_seen

def trainer(train_state: TrainState,data_loaders: DataLoaders, training_config: TrainConfig):

    model,checkpoint_path = train_state.model, train_state.checkpoint_path
    _,training_steps,_,_,learning_rate,device,_,force = astuple(training_config)
    
    warmup_steps = 100    
    min_lr = 3e-5           

    torch.manual_seed(42)

    print(f"Using {device}")

    optimizer = torch.optim.AdamW(model.parameters(),learning_rate, weight_decay=0.1)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=training_steps - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
    
    if os.path.exists(checkpoint_path) and not force:
        model,optimizer,scheduler = load_model(model,inference=False,checkpoint_path=checkpoint_path,
                                               device=device,optimizer=optimizer,scheduler=scheduler)
        
    train_state.optimizer = optimizer
    train_state.scheduler = scheduler
    train_state.model = model

    start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model(
        train_state,data_loaders,training_config
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.") 
    return train_losses,val_losses,tokens_seen
