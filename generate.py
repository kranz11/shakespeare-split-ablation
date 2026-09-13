import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_batch, load_data, chunk_split, tail_split
from config import *
from model import SmallLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer, data = load_data(data_path)
print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")

model = SmallLM(
    vocab_size=tokenizer.vocab_size,
    input_dim=input_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size,
    dropout=dropout,
).to(device)

checkpoint  = torch.load(inference_path)
model.load_state_dict(checkpoint["model_state_dict"])
tokenizer.stoi = checkpoint["stoi"]
tokenizer.itos = checkpoint["itos"]
model.eval()

start_ids = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=device)
generated = model.generate(start_ids, max_new_tokens=300, temperature=0.5, top_k=10)
print(tokenizer.decode(generated[0].tolist()))
