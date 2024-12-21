import os
import requests
import numpy as np
from datasets import load_dataset
import tiktoken

# Load dataset
ds = load_dataset("roneneldan/TinyStories")
enc = tiktoken.get_encoding("gpt2")

# Define the output directory
output_directory = '/home/jzhangln/nanoGPT/data/tinystories/'

def flatten(matrix):
    return [item for row in matrix for item in row]  # flatten the list of lists

# Tokenize validation data
val_ids = [enc.encode_ordinary(i) for i in ds['validation']['text']]
val_ids = flatten(val_ids)
val_ids = np.array(val_ids, dtype=np.uint16)

# Save validation data
val_output_path = os.path.join(output_directory, 'val.bin')
val_ids.tofile(val_output_path)

# Define output path for training data
output_path = os.path.join(output_directory, 'train.bin')

def chunk_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        chunk = input_list[i:i + chunk_size]
        yield [enc.encode_ordinary(item) for item in chunk]  # divide train data into chunks and tokenize it

# Save training data
with open(output_path, 'wb') as f:
    for encoded_chunk in chunk_list(ds['train']['text'], 1000):
        flattened_chunk = [item for sublist in encoded_chunk for item in sublist]  # flatten the list of lists
        train_ids = np.array(flattened_chunk, dtype=np.uint16)
        train_ids.tofile(f)  # Save to the opened file object

print("Finish'")
