import sys
import torch
from utils import clear_gpu_memory

def cross_entropy_loss_batch(model, input_batch, target_batch, device):

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # Forward pass
    logits = model(input_batch)
    del input_batch

    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_batch.reshape(-1),
        ignore_index= -100
    )

    del target_batch, logits
    return loss


def calculate_loss(model, input_batch, target_batch, device):

    loss = cross_entropy_loss_batch(model,input_batch,target_batch,device)
    return loss

def calc_loader_loss(model, data_loader, device, global_step, val_loss_file, num_batch=None, type_train=True):
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
            loss = calculate_loss(model,input_batch,target_batch,device)

            if not type_train:
                progress = (i + 1) / num_batch
                sys.stdout.write(
                    f"\r[Eval {i+1:4d}/{num_batch}]  "
                    f"Progress: {progress*100:6.2f}%  "
                    f"Step Loss: {global_step}, {loss}  "
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

def evaluate_model(model, tokenizer, val_loader, device, eval_iter, global_step, val_loss_file):
    model.eval()
    with torch.no_grad():

        val_loss = calc_loader_loss(model, val_loader, device, global_step, val_loss_file, num_batch=eval_iter, type_train=False)
        
        prompt = tokenizer.encode("Selam kimsin?", return_tensors="pt")
        prompt = prompt.to(device)

        generated_tokens = model.module.generate(prompt)
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
    model.train()
    return val_loss, generated_text
