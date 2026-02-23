import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


import time, os, sys
from dataclasses import astuple
from tqdm import tqdm
import math

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import (
    TrainState,
    TrainConfig,
    DataLoaders,
    clear_gpu_memory,
    write_norm_to_file
)

from .loss_func import(
    calculate_loss,
    evaluate_model
)

class Trainer:

    def __init__(self, train_state: TrainState,data_loaders: DataLoaders, training_config: TrainConfig, logger, hub_info):
        
        self.train_state = train_state
        self.data_loaders = data_loaders
        self.training_config = training_config
        self.logger  =logger
        self.hub_info = hub_info

    def train(self):
        train_losses, val_losses, tokens_seen = self._train(
            train_state=self.train_state,
            data_loaders=self.data_loaders,
            training_config=self.training_config, 
            logger=self.logger,
            hub_info = self.hub_info
        )
        return train_losses,val_losses,tokens_seen
    
    def _train(self,train_state: TrainState, data_loaders: DataLoaders, training_config: TrainConfig, logger, hub_info):
        start_time = time.time()

        model = train_state.model
        checkpoint_path = train_state.checkpoint_path
        tokenizer = train_state.tokenizer

        train_loader = data_loaders.train
        val_loader = data_loaders.val

        num_epochs = training_config.num_epochs
        training_steps = training_config.training_steps
        eval_steps = training_config.eval_steps
        eval_sample = training_config.eval_sample
        learning_rate = training_config.learning_rate
        device = training_config.device
        output_path = training_config.output_path
        force = training_config.force

        train_loss_file = f"{output_path}/train_loss.txt"
        val_loss_file = f"{output_path}/validation_loss.txt"
        generated_text_file = f"{output_path}/validation_loss.txt"


        gpu_id = device
        model = DDP(model,device_ids=[gpu_id])
        del train_state, data_loaders, training_config

        warmup_steps = 100
        min_lr = 3e-5

        optimizer = torch.optim.AdamW(model.parameters(),learning_rate, weight_decay=0.1)
        scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        scheduler_decay = CosineAnnealingLR(optimizer, T_max=training_steps - warmup_steps, eta_min=min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
        if checkpoint_path  and os.path.exists(checkpoint_path) and not force and gpu_id == 0:
            model = model.module.from_pretrained(checkpoint_path)

            logger.info(f"GPU {gpu_id}: Resuming training from snapshot.")

        train_losses, val_losses, track_tokens_seen = [],[],[]
        token_seen , global_step = 0, -1

        best_val_loss = float("inf")
        total_tokens = 0
        for epoch in range(num_epochs):
            model.train()
            torch.autograd.set_detect_anomaly(True)
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if gpu_id == 0 else train_loader
            for input_batch, target_batch in pbar:

                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                torch.autograd.set_detect_anomaly(True)
                loss = calculate_loss(model,input_batch,target_batch,device)

                if loss is None:
                    logger.warning(f"at step {global_step} train loss is None")
                    continue
                    
                loss.backward()

                global_step += 1
                write_norm_to_file(model=model, num_layers=model.module.config.num_hidden_layers,global_step=global_step, out_path=output_path)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                

                if gpu_id == 0:
                    pbar.set_postfix({"train loss":f"{loss.item():.4f}"})
                    with open(train_loss_file, "a") as f:
                        f.write(f"{global_step}, {loss.item()}\n")

                if  (global_step > 0 and global_step % eval_steps == 0):
                    if gpu_id == 0:
                        print("Calculating loss...", end="", flush=True)

                        val_loss, generated_text = evaluate_model(
                            model, tokenizer, val_loader, device, eval_sample, val_loss_file
                        )
                        logger.info(f"Model generated text: {generated_text}")
                        with open(generated_text_file, "a") as f:
                            f.write(f"{global_step}, {generated_text}\n")
                        val_losses.append(val_loss)
                        ppl = math.exp(val_loss)
                        total_tokens += input_batch.numel()
                        sys.stdout.write("\r")
                        sys.stdout.write(" " * 50 + "\r")
                        sys.stdout.flush()
                        logger.info(f"Ep {epoch+1} (Step {global_step:06d}): "
                            f"Train loss {loss:.3f} Val loss {val_loss:.3f}, Perplexity {ppl:.3f}")
                        
                        model.push_to_hub(hub_info.repo_name,
                                          commit_message=f"Ep {epoch+1} (Step {global_step:06d}): "
                            f"Train loss {loss:.3f} Val loss {val_loss:.3f}, Perplexity {ppl:.3f}")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            model.module.save_pretrained(f"{output_path}/{model.module.name}_best_model")
            clear_gpu_memory()

        end_time = time.time()

        execution_time_minutes = (end_time - start_time) / 60
        if gpu_id == 0:
            logger.info(f"Number of Train Tokens: {total_tokens}")
            logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")
        return train_losses, val_losses, track_tokens_seen