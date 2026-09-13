import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Chartokenizer(nn.Module):
    def __init__(self,text:str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int -> string

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """
    Randomly samples `batch_size` chunks of length `block_size` from
    `data`, along with the corresponding "shifted by one" targets used
    for next-token-prediction training.
    """
    # Pick random starting indices such that we never run off the end.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

def load_data(data_path: str):
    """
    Reads the source text file, builds a Chartokenizer from it, and
    encodes the full text as a tensor of token ids.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Chartokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return tokenizer, data

def chunk_split(block_size,data):
    chunk_size = block_size * 4  # size of the units we shuffle; bigger than block_size so most training sequences still come from one contiguous, coherent passage
    num_chunks = len(data) // chunk_size  # how many whole chunks fit in the data
    data = data[: num_chunks * chunk_size]  # drop the few leftover tokens that don't fill a final full chunk, so we can reshape evenly

    chunks = data.view(num_chunks, chunk_size)  # reinterpret the flat token stream as (num_chunks, chunk_size) -- each row is one contiguous slice of the original text

    split_rng = torch.Generator().manual_seed(42)  # separate, fixed-seed RNG just for the split, so the same chunks land in train/val on every run regardless of training randomness
    shuffle_order = torch.randperm(num_chunks, generator=split_rng)  # a random permutation of chunk indices, deterministic because of the seeded generator above
    chunks = chunks[shuffle_order]  # reorder the chunks according to that random permutation

    n_val_chunks = max(1, int(0.1 * num_chunks))  # how many chunks (at least 1) to hold out for validation -- roughly 10% of the data
    val_data = chunks[:n_val_chunks].reshape(-1)  # take the first n_val_chunks post-shuffle chunks and flatten them back into a single 1D token stream
    train_data = chunks[n_val_chunks:].reshape(-1)  # the remaining, larger group of chunks becomes the training stream
    return train_data,val_data

def tail_split(data):
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data,val_data