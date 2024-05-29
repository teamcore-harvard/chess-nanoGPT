# %%
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets
import pickle
from collections import defaultdict
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 14
dtype = np.uint8  # Currently there are only 32 tokens in the chess LLMs vocab
training_ranges = [[600, 1100], [1100, 1500], [1500, 1900]]

# %%
dataset_path = "adamkarvonen/chess_games"
file_path = "lichess_200k_elo_bins.zip"
# file_path = "smaller_pgn_file_blocks.zip"

# Load the dataset
dataset = load_dataset(dataset_path, data_files=file_path)

# %%
dataset['train'][-1]

# %%
elo_bins = dataset['train']['elo_bin']
bins = set(elo_bins)
bins

# %%

elo_bin_idx = defaultdict(list)
for idx, elo in enumerate(elo_bins):
    elo_bin_idx[elo].append(idx)

# %%

# %%
training_range_idx = defaultdict(list)
for tr in training_ranges:
    for b in bins:
        bin_range = eval(b.replace('[', '('))
        if bin_range[0] >= tr[0] and bin_range[0] < tr[1]:
            training_range_idx[str(tr)] += (elo_bin_idx[b])

for t in training_range_idx:
    print(t, 'has number of games: ', len(training_range_idx[t]))

# %%
datasets = [dataset['train'].select(training_range_idx[t]) for t in training_range_idx]

# %%
for idx, t in enumerate(training_ranges):
    dataset = datasets[idx]
    # by default only contains the 'train' split, so create a test split
    split_dataset = dataset.train_test_split(
        test_size=0.01, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. Using meta.pkl in the same directory as this file
    meta_path = os.path.join(os.path.dirname(__file__), "meta.pkl")
    # meta_path = '/home/ezipe/git/chess_transformer_mothership/chess-nanoGPT/data/lichess_hf_dataset/meta.pkl'
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    itos = meta["itos"]

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint8, mode='r')
    # print(split_dataset["val"][0])
    # print(len(split_dataset["val"]["transcript"][0]))

    # For verifying that all games are 1024 tokens long
    # for game in split_dataset["train"]["transcript"]:
    #     if len(game) != 1024:
    #         print(len(game))
    #         print(game)
    #         break
    # print(stoi)

    column_name = "transcript"

    def process(example):
        ids = np.array([stoi[c] for c in example[column_name]], dtype=dtype)
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=[column_name],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # print(tokenized["val"]["ids"])

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len} tokens")
        # dname = '/home/ezipe/git/chess_transformer_mothership/chess-nanoGPT/data/lichess_hf_dataset/'
        dname = os.path.dirname(__file__)
        filename = os.path.join(dname, f"{split}_{t[0]}_{t[1]}.bin")
        
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        print(arr.shape)
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            # print(batch[0])
            arr_batch = np.concatenate(batch["ids"])
            # print(arr_batch)
            # print(arr_batch.shape)
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()



