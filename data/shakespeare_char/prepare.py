import os
import pickle
import requests
import numpy as np
from tqdm import tqdm

def download_data(file_path: str, url: str):
    """Download data from the provided URL to the specified file path."""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(requests.get(url).text)

def build_vocab_set_and_file_length(input_file: str, chunk_size: int = int(1e5)) -> (set, int):
    """Build a set of unique characters from the entire file and calculate file length."""
    unique_chars = set()
    file_length = 0
    with open(input_file, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            unique_chars.update(chunk)
            file_length += len(chunk)

    return unique_chars, file_length


def get_mappings(chars: set) -> (set, dict, dict):
    """Return mappings based on the set of characters."""
    chars = sorted(list(chars))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos

def encode_chunk(chunk: str, stoi: dict) -> list:
    """Encode a chunk of text using the stoi mapping."""
    return [stoi[c] for c in chunk]

def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


def process_data_in_chunks(input_file: str, output_dir: str, stoi: dict, file_length: int, block_size: int = 1024, chunk_size: int = 10000, split_ratio: float = 0.98):
    """Process the dataset in chunks to manage memory usage."""
    
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')

    # Calculate the split point based on the entire file length
    split_point = int(file_length * split_ratio)
    split_point -= split_point % block_size

    # Initialize counters
    data_size_processed = 0
    train_tokens_count = 0
    val_tokens_count = 0

    num_chunks = -(-file_length // chunk_size)  # Ceiling division
    with tqdm(total=num_chunks, desc="Processing Chunks") as pbar:
        with open(input_file, 'r') as f, open(train_file, 'wb') as train_f, open(val_file, 'wb') as val_f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                encoded_chunk = encode_chunk(chunk, stoi)
                chunk_length = len(encoded_chunk)
                data_size_processed += chunk_length
                dtype=np.uint8

                # Determine if the chunk crosses the split point
                if data_size_processed < split_point:
                    train_chunk = np.array(encoded_chunk, dtype=dtype)
                    train_chunk.tofile(train_f)
                    train_tokens_count += chunk_length
                elif data_size_processed - chunk_length < split_point:
                    # Split the chunk into train and val
                    train_end = split_point - (data_size_processed - chunk_length)
                    train_chunk = np.array(encoded_chunk[:train_end], dtype=dtype)
                    val_chunk = np.array(encoded_chunk[train_end:], dtype=dtype)
                    train_chunk.tofile(train_f)
                    val_chunk.tofile(val_f)
                    train_tokens_count += train_end
                    val_tokens_count += chunk_length - train_end
                else:
                    val_chunk = np.array(encoded_chunk, dtype=dtype)
                    val_chunk.tofile(val_f)
                    val_tokens_count += chunk_length

                # Update the progress bar
                pbar.update(1)

    print(f"train has {train_tokens_count:,} tokens")
    print(f"val has {val_tokens_count:,} tokens")
    print(f"Processed {data_size_processed:,} characters.")

    print(f"train / blocksize = {train_tokens_count / block_size}")
    print(f"val / blocksize = {val_tokens_count / block_size}")
    print(f"total / block_size = {data_size_processed / block_size}")

# Usage

# directory_path = os.getcwd() # use this for jupyter notebooks
directory_path = os.path.dirname(__file__)
input_file_path = os.path.join(directory_path, 'input.txt')
output_dir = directory_path
data_url = 'https://adam-karvonen-chess.s3.us-east-2.amazonaws.com/5gb_batch_input.txt'
download_data(input_file_path, data_url)
# process_data_in_chunks(input_file_path, os.path.dirname(__file__))

block_size = 1024
unique_chars, file_length = build_vocab_set_and_file_length(input_file_path, int(1e5))
print(f"length of dataset in characters: {file_length:,}")
print(f"dataset length / block_size = {file_length/block_size}")

chars, stoi, itos = get_mappings(unique_chars)

vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# Saving metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

process_data_in_chunks(input_file_path, output_dir, stoi, file_length, chunk_size=int(1e6))