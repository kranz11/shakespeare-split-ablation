input_dim = 256        # size of each token's embedding vector
num_heads = 8           # number of parallel attention heads per block
num_layers = 8          # number of stacked TransformerBlocks
block_size = 256        # max context length (in tokens) the model can see at once
batch_size = 32         # number of sequences processed per training step
dropout = 0.1            # dropout probability, helps prevent overfitting
lr = 3e-4                # learning rate for the AdamW optimizer
max_iters = 6000         # total number of training steps
eval_interval = 200      # how often (in steps) to print train/val loss
out_dir = 'checkpoints'  # folder where the trained model checkpoint is saved
log_dir = 'logs'  # folder where per-run training logs are saved

data_path = 'shakespeare.txt'  # source text file to train on
split_method = "chunk"  # "chunk" -> chunk_split (shuffled contiguous chunks), "tail" -> tail_split (last 10% as val)

inference_path = "model.pt" # path to model.pt
