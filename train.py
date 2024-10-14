"""

main training script.

doesnt support :
-multi GPU

when launching a training run, will create a name for the run (wandb run or random if no wandb logging) and save checkpoint, config and final model in runs/{run_name}

if you're using the WSD scheduler and you just want to cooldown a model over N steps, set :
lr_warmup_iters = 0
lr_decay_iters = N
num_iters = N

also, when using the WSD scheduler, a checkpoint will automatically be saved just before the cooldown (independently of ckpt_interval)

"""

import os
import string
from contextlib import nullcontext
from dataclasses import asdict
import json
import random
import math
import numpy as np
import time
import wandb

import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch._inductor.config as torch_ind_config

from utils.lr_schedules import cosine_warmup_schedule, wsd_schedule

from models.lm import LM
from data.dataloader import DataLoader
from utils.misc import format_time
from config import *

seed = 123456789 + seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"
torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
dtype_ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, torch_dtype))

if log_wandb:
    wandb.init(project="arena",
            config={
                "data": {
                    "ctx_len": ctx_len,
                    "vocab_size": vocab_size,
                },
                "model": {
                    "architecture": architecture,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "bias": bias,
                    "base_std": base_std,
                    # Transformer
                    "d_ff": d_ff,
                    "diff_transformer": diff_transformer,
                    "n_heads": n_heads,
                    "n_kv_heads": n_kv_heads,
                    "dropout": dropout,
                    "pos_emb": pos_emb,
                    "rope_theta": rope_theta,
                    # Mamba2
                    "d_head_m2": d_head,
                    "d_state_m2": d_state,                    
                },
                "training": {
                    "seed": seed-123456789,
                    "num_iters": num_iters,
                    "total_batch_size": total_batch_size,
                    "micro_batch_size": micro_batch_size,
                    "optimizer": optimizer,
                    "adam_b1": adam_b1,
                    "adam_b2": adam_b2,
                    "max_grad_norm": max_grad_norm,
                    "weight_decay": weight_decay,
                    # lr
                    "schedule": schedule,
                    "lr": lr,
                    "lr_min": lr_min,
                    "lr_warmup_iters": lr_warmup_iters,
                    "lr_decay_iters": lr_decay_iters,
                    # muP
                    "use_mup": use_mup,
                    "mup_base_width": mup_base_width,
                    # ademamix
                    "adam_b3": adam_b3,
                    "alpha": alpha,
                    
                }
            })

if log_wandb:
    run_name = wandb.run.name
else:
    run_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))

save_dir = os.path.join(save_dir, run_name)
os.makedirs(save_dir, exist_ok=True)
print(f"Run name: {run_name}.")

train_loader = DataLoader("data/fineweb10B/fineweb_train_*.bin", micro_batch_size, ctx_len, 1, 1)
val_loader = DataLoader("data/fineweb10B/fineweb_val_*.bin", micro_batch_size, ctx_len, 1, 1)

tokens_per_iter = total_batch_size * ctx_len
grad_acc_steps = total_batch_size // micro_batch_size

print(f"Tokens processed per iteration: {tokens_per_iter}")
print(f"Number of micro batches: {grad_acc_steps}")

# model
config_dict = asdict(config) # config is defined in config.py
config_dict['architecture'] = architecture
json.dump(config_dict, open(os.path.join(save_dir, 'config.json'), 'w'))

g = torch.Generator()
g.manual_seed(seed)

model = LM(config, vocab_size=vocab_size, rng=g).to(device)

optim = model.configure_optimizers(optimizer, weight_decay, lr, (adam_b1, adam_b2), device_type, beta3=adam_b3, alpha=alpha, T_ab3=num_iters)

if ckpt != "":
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optimizer"])

if schedule == "cosine":
    scheduler = lr_scheduler.LambdaLR(optim, cosine_warmup_schedule(lr=lr, lr_min=lr_min, warmup_iters=lr_warmup_iters, num_iters=num_iters, start_iter=start_iter))
elif schedule == "wsd":
    scheduler = lr_scheduler.LambdaLR(optim, wsd_schedule(warmup_iters=lr_warmup_iters, decay_iters=lr_decay_iters, num_iters=num_iters, start_iter=start_iter))
else:
    raise NotImplementedError

num_params = sum([p.numel() for p in model.parameters()])
print(f"Model initialized. Number of parameters : {num_params}.")

unoptimized_model = model # the unoptimized model is kept for saving
if use_torch_compile:
    if hasattr(torch_ind_config, "coordinate_descent_tuning"):
        torch_ind_config.coordinate_descent_tuning = True

    model = torch.compile(model)

print("Training is starting.")

start_time = time.time()
torch.cuda.reset_peak_memory_stats()

try:
    for iter in range(start_iter, num_iters):
        t0 = time.time()

        loss_total = 0.
        for micro_step in range(grad_acc_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with dtype_ctx:
                loss = model(x, y)
                loss = loss / grad_acc_steps
                loss_total += loss.detach()

            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optim.step()
        optim.zero_grad(set_to_none=True)

        model.norm_weights()

        t1 = time.time()

        # lr decay
        scheduler.step()
        lr_iter = scheduler.get_last_lr()[1] # param group 1 has a "fixed" lr (ie not affected by muP)
        
        # val loss
        if (iter % eval_val_interval == 0):
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                for i in range(eval_val_iters):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                    with dtype_ctx:
                        loss = model(x, y)
                    eval_loss += loss.item()

                eval_loss /= eval_val_iters
                model.train()

        # checkpointing
        if (ckpt_interval and iter % ckpt_interval == 0 and iter > 0) or (schedule == "wsd" and (iter == num_iters-lr_decay_iters)):
            
            dirname = f"ckpt_{iter}/"
            if (schedule == "wsd" and (iter == num_iters-lr_decay_iters)):
                print("----- Starting cooldown -----")
                dirname = f"ckpt_{iter}_before_cooldown"

            os.makedirs(os.path.join(save_dir, dirname), exist_ok=True)
            checkpoint = {"model": unoptimized_model.state_dict(),
                          "optimizer": optim.state_dict()}
            torch.save(checkpoint, os.path.join(save_dir, dirname, "model.pth"))

        # logging : print and wandb
        to_log = {}
        if iter % train_log_interval == 0:
            tokens_per_s = tokens_per_iter / (t1 - t0)
            to_log.update({"train_loss": loss_total, "grad_norm": norm})
            to_log.update({"tokens_per_s": tokens_per_s})

        if iter % eval_val_interval == 0:
            to_log.update({"val_loss": eval_loss})
            
        if to_log:
            tokens_seen = (iter+1)*ctx_len*total_batch_size
            to_log.update({"lr": lr_iter, "tokens_seen": tokens_seen})

            # printing
            if "val_loss" in to_log:
                num_digits = len(str(num_iters))
                formatted_iter = f"{iter:0{num_digits}d}"

                uptime = time.time() - start_time
                total_time = ((num_iters-start_iter) * uptime) / (iter-start_iter) if iter>start_iter else -1
                eta = total_time - uptime

                print(f"Iter {formatted_iter}/{num_iters}. train loss: {loss_total:.3f}. valid loss: {eval_loss:.3f}. lr: {lr_iter:.5f}. {tokens_per_s:.0f} tok/s. uptime: {format_time(uptime)}. ETA: {format_time(eta)}")
            
            # logging
            if log_wandb:
                wandb.log(to_log, step=iter)
        
        # skip first step for timing calculations (act as a warm-up)
        if iter==start_iter:
            start_time = time.time()
        
except KeyboardInterrupt:
    print("Training interrupted.")

end_time = time.time()
print(f"Training is done. Took {(end_time-start_time)/60:.2f} minutes.")

# saving model checkpoint (model+optim)
checkpoint = {"model": unoptimized_model.state_dict(),
              "optimizer": optim.state_dict()}
torch.save(checkpoint, os.path.join(save_dir, "model.pth"))

print(f"Successfully saved checkpoint and config in {save_dir}.")

# final logging (some metrics for wandb)

#todo : réfléchir à d'autres
to_log = {"num_params": num_params,
          "num_iters_done": iter,
          "num_tokens": iter*total_batch_size*ctx_len,
          "use_torch_compile": use_torch_compile,
          "use_flash_attn": use_flash_attention,
          "peak_memory": torch.cuda.max_memory_allocated() // 1024 // 1024, # MiB
          "tokens_per_sec": tokens_per_s,
          "dtype": dtype}

if log_wandb:
    wandb.log(to_log)
    wandb.finish()
