import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_batch, load_data, chunk_split, tail_split
from config import *
from model import SmallLM

os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"train_{split_method}_{datetime.now():%Y%m%d_%H%M%S}.log")

def log(msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Using device: {device}")

tokenizer, data = load_data(data_path)
log(f"Vocabulary size: {tokenizer.vocab_size} unique characters")

if split_method == "chunk":
    train_data, val_data = chunk_split(block_size, data)
elif split_method == "tail":
    train_data, val_data = tail_split(data)
else:
    raise ValueError(f"Unknown split_method: {split_method!r} (expected 'chunk' or 'tail')")

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    """Averages the loss over several random batches for a more stable
    estimate than a single batch would give (mainly used for logging)."""
    model.eval()
    out = {}
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out

def save_checkpoint(val_loss):
    # Called only when we've just seen a new best val loss, so
    # model.pt always holds the best-generalizing checkpoint seen
    # during training rather than whatever the weights happen to
    # look like at the final step (which may already be overfit).
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "val_loss": val_loss,
            "config": {
                "input_dim": input_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "block_size": block_size,
                "batch_size": batch_size,
                "dropout": dropout,
                "lr": lr,
                "max_iters": max_iters,
            },
        },
        checkpoint_path,
    )

# ---- Build the model -------------------------------------------------
model = SmallLM(
    vocab_size=tokenizer.vocab_size,
    input_dim=input_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size,
    dropout=dropout,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
log(f"Model has {num_params:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

os.makedirs(out_dir, exist_ok=True)
checkpoint_path = os.path.join(out_dir, split_method, "model.pt")

best_val_loss = float("inf")

# ---- Training loop ---------------------------------------------------
for step in range(max_iters):
    # Periodically evaluate on held-out data so we can see whether
    # the model is generalizing or just memorizing the training set.
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(
            model, train_data, val_data, block_size, batch_size, device
        )
        log(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            save_checkpoint(best_val_loss)
            log(f"  -> new best val loss {best_val_loss:.4f}, checkpoint saved")

    xb, yb = get_batch(train_data, block_size, batch_size, device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

log(f"\nBest val loss: {best_val_loss:.4f}")
log(f"Best checkpoint saved to {checkpoint_path}")
