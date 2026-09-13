import os

# ---- model ----
input_dim   = 256        # size of each token's embedding vector
num_heads   = 8          # number of parallel attention heads per block
num_layers  = 8          # number of stacked transformer decoder blocks
block_size  = 256        # max context length (in tokens) the model can see at once
dropout     = 0.1        # dropout probability

# ---- training ----
batch_size    = 32       # sequences per training step
lr            = 3e-4     # AdamW learning rate
max_iters     = 6000     # total training steps
eval_interval = 200      # how often (in steps) to evaluate train/val loss

# ---- the experiment ----
# "chunk" -> chunk_split: shuffle contiguous 4*block_size chunks, hold out 10%
# "tail"  -> tail_split:  hold out the last 10% of the text
# This is the only thing that differs between the two runs in the README.
split_method = "chunk"

# ---- paths (relative, so the repo runs anywhere) ----
data_path = "shakespeare.txt"
out_dir   = "checkpoints"
log_dir   = "logs"

# Checkpoints are written per split so the two runs don't overwrite each other.
inference_path = os.path.join(out_dir, split_method, "model.pt")
