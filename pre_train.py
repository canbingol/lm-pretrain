import time
from itertools import islice

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import tqdm

from utils import clear_gpu_memory, load_model

def calcc(model,input_batch,target_batch,device):
    total_loss = 0
    for i in range(len(input_batch)):
        input = input_batch.to(device, non_blocking=True)
        logits = model(input)
        del input
        target = target_batch.to(device, non_blocking=True)

        logits = logits.reshape(logits.shape[0] * logits.shape[1],-1)
        target = target.reshape(target.shape[0] * target.shape[1])

        loss = torch.nn.functional.cross_entropy(logits,target)
        total_loss += loss
        del target,logits,loss
    
    clear_gpu_memory()
    return total_loss / len(input_batch)


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


def train_model(model,optimizer,scheduler,train_loader,val_loader,num_epochs,device,eval_steps,training_steps,eval_sample,output_path):

    train_losses, val_losses, track_tokens_seen = [],[],[]
    token_seen , global_step = 0, -1

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in tqdm.tqdm(islice(train_loader,0,training_steps), total=training_steps):

            loss = calcc_loss_batch(model,input_batch,target_batch,device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            token_seen = input_batch.numel()
            global_step += 1


            if global_step % eval_steps == 0:
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


def trainer(model,train_loader,val_loader,num_epochs,training_steps,eval_steps,eval_sample,learning_rate,device,output_path,checkpoint_path,force):
    warmup_steps = 100    
    min_lr = 3e-5           


    torch.manual_seed(42)

    print(f"Using {device}")

    optimizer = torch.optim.AdamW(model.parameters(),learning_rate, weight_decay=0.1)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=training_steps - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
    
    if checkpoint_path and not force:
        model,optimizer,scheduler = load_model(model,False,checkpoint_path,
                                               device,optimizer,scheduler)
        start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model(
        model,optimizer,scheduler, train_loader, val_loader,
        num_epochs, device,eval_steps, training_steps,eval_sample,output_path
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.") 
    return train_losses,val_losses,tokens_seen