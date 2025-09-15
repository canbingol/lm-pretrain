import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import islice
from datetime import datetime
from huggingface_hub import login
from huggingface_hub import HfApi


def eval_model(model,val_loader,criterion,DEVICE,LOGGING,MAX_EVAL_STEP,val_loss_file,global_val_step):
    model.eval()
    step_loss = 0
    count = 0
    for i,(input_batch,target_batch) in enumerate(val_loader):
        if LOGGING:
            print(f"{i}th eval step progrss...")

        input_batch = input_batch.to(DEVICE)
        target_batch = target_batch.to(DEVICE)
        with torch.no_grad():
            logits = model(input_batch)
            logits = logits.to(DEVICE)
        
        logits = logits.reshape(logits.shape[0] * logits.shape[1],-1)
        target_batch = target_batch.reshape(target_batch.shape[0] * target_batch.shape[1])

        loss = criterion(logits,target_batch,ignore_index=0)
        with open (val_loss_file,"a") as f:
            f.write(f"{global_val_step+1} step validation loss : {loss.item()}\n")
        step_loss += loss.item()
        count += 1
        if i >= MAX_EVAL_STEP:
            break
    model.train()
    return step_loss / count, count,global_val_step

def push_to_hub(api, model_path):
    print("Pushing to hub model")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path), 
        repo_id="canbingol/qwen3-tr",
        repo_type="model",
    )


def lm_pretrain(model,train_loader,val_loader,criterion,optimizer,EPOCH,MAX_TRAIN_STEP,MAX_EVAL_STEP,LOGGING,DEVICE,SAVE_STEP,OUTPUT_PATH,MODEL_INFO,cpt_epoch=0,cpt_step=0):

    login(os.getenv("HF_TOKEN"))
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file = os.path.join(OUTPUT_PATH, "logs", f"log_{timestamp}/log.txt")
    val_loss_file = os.path.join(OUTPUT_PATH, "logs", f"log_{timestamp}/val_loss.txt")
    train_loss_file = os.path.join(OUTPUT_PATH, "logs", f"log_{timestamp}/train_loss.txt")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    def model_info():
        num_params = sum(p.numel() for p in model.parameters())
        param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        size_mb = param_size_bytes / (1024**2)
        size_gb = param_size_bytes / (1024**3)

        print(f"Toplam Parametre Sayısı: {num_params:,}")
        print(f"Yaklaşık Boyut: {size_mb:.2f} MB ({size_gb:.2f} GB)")

    if MODEL_INFO:
        model_info()

    model.train()
    step = cpt_step
    global_val_step = 0
    FIRST = True

    for epoch in range(cpt_epoch,EPOCH):
        start_time = datetime.now()
        start_in_epoch = (step % MAX_TRAIN_STEP)    
        batch_count = 0 
        val_batch_count = 0
        epoch_train_loss = 0
        epoch_val_loss = 0
        train_steps_loss = 0

        print(f"+--------------------------------- {epoch+1} EPOCH ------------------------------------+")
        bar = tqdm(islice(train_loader, start_in_epoch,MAX_TRAIN_STEP), total=MAX_TRAIN_STEP,initial=start_in_epoch,
            desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch")
        for i,(input_batch, output_batch) in enumerate(bar):
            current_step = step + 1
            batch_count += 1    
            if LOGGING:
                print(f"{epoch}th epoch {i}th training step progrss...")
            input_batch = input_batch.to(DEVICE,non_blocking=True)
            output_batch = output_batch.to(DEVICE,non_blocking=True)

            logits = model(input_batch)
            logits = logits.to(DEVICE)
            
            optimizer.zero_grad()
            logits = logits.reshape(logits.shape[0] * logits.shape[1], -1)
            output_batch = output_batch.reshape(output_batch.shape[0] * output_batch.shape[1])

            if LOGGING:
                print(f"training logits shape {logits.shape}\n training target shape {output_batch.shape}")

            loss = F.cross_entropy(logits, output_batch, ignore_index=0)
            with open (train_loss_file,"a") as f:
                f.write(f"{step+1} step train loss : {loss.item()}\n")
            epoch_train_loss += loss.item()
            train_steps_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if current_step % SAVE_STEP == 0 and step != 0:
                save_path = os.path.join(OUTPUT_PATH, f"checkpoint_epoch{epoch+1}_step{step}.pt")
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }, save_path)
                print(f"[INFO] Model saved at {save_path}")
                push_to_hub(api,save_path)
            bar.set_postfix(train_loss=loss.item())
            if (current_step % MAX_EVAL_STEP == 0 or current_step == MAX_TRAIN_STEP) and step != 0:

                eval_loss,count,global_val_step = eval_model(model,val_loader,criterion,DEVICE,LOGGING,MAX_EVAL_STEP,val_loss_file,global_val_step)
                if FIRST :
                    epoch_val_loss += eval_loss
                    FIRST = False
                else:
                    epoch_val_loss += eval_loss
                    epoch_val_loss /= 2
                val_batch_count += count
                print(f"{step}.th step | train-loss: {train_steps_loss:.5f} | eval-loss: {epoch_val_loss:.5f}")
                train_steps_loss = 0.0    
            step += 1
        step = current_step
        end_time = datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        save_path = os.path.join(OUTPUT_PATH, f"checkpoint_epoch{epoch+1}_final.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": (epoch_train_loss / max(1, batch_count)),  
        }, save_path)
        push_to_hub(api,save_path)
        print("\n================ EPOCH SUMMARY ================")
        print("Epoch        ||   Train Loss       ||   Val Loss   ||   Duration   ||   Save Path")
        print("---------------------------------------------------------------")
        print(f"{epoch+1:<5} || {epoch_train_loss / max(1, batch_count):<13.6f}   || {epoch_val_loss :<11.6f}   || {duration_str:<12}|| {save_path}")
        print("================================================\n")

        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        row = f"{epoch+1:<5} || {epoch_train_loss / max(1, batch_count):<13.6f}   || {epoch_val_loss:<11.6f}     || {duration_str:<12}|| {save_path}\n"

        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("Epoch ||   Train Loss      ||   Val Loss   ||   Duration   ||   Save Path\n")
                f.write("---------------------------------------------------------------\n")

        with open(log_file, "a") as f:
            f.write(row)

        if epoch+1 == EPOCH:  
            with open(log_file, "r") as f:
                print("\n================ TRAINING LOG ================\n")
                print(f.read())
                print("=============================================\n")

