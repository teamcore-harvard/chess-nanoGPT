# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-1500_1900'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'chess-gpt-batch'
wandb_run_name = 'chess_1500-1900_' + time.strftime("%Y-%m-%d_%H-%M-%S")

dataset = 'lichess_hf_dataset'
gradient_accumulation_steps = 1
batch_size = 120
block_size = 1024

# baby GPT model :)
n_layer = 16
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 3e-4 # with baby networks can afford to go a bit higher
max_iters = 5116 # 5116 * 1024 * 120 ~= 628,683,781 (1500-1900 Elo dataset size)
lr_decay_iters = 5116 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 50 # not super necessary potentially
compile = True

low_elo = 1500
high_elo = 1900


# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
