"""
STEP 1: Prepare the data.

A neural network can't read letters directly — it only understands numbers.
So this script:
  1. Generates a Q&A dataset (Question/Answer pairs in plain English) via
     make_qa_dataset.py, teaching the model conversation structure instead
     of open-ended prose
  2. Finds every unique character used in it
  3. Builds a "translation table" between characters and numbers
  4. Saves the whole text as a big list of numbers, ready for training

Run this first:  python prepare_data.py
"""

import os
import subprocess
import numpy as np
import pickle

# 1. Generate the Q&A dataset if we don't already have it
input_file = "qa_data.txt"

if not os.path.exists(input_file):
    print("Generating Q&A dataset...")
    subprocess.run(["python3", "make_qa_dataset.py"], check=True)

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total characters in dataset: {len(text):,}")

# 2. Find every unique character (this becomes our "vocabulary")
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique characters (vocab size): {vocab_size}")
print(f"They are: {''.join(chars)!r}")

# 3. Build character <-> number lookup tables
stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
itos = {i: ch for i, ch in enumerate(chars)}  # int to string

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

# 4. Convert the entire text into numbers
data = np.array(encode(text), dtype=np.uint16)

# Split into training data (90%) and validation data (10%)
# Validation data lets us check the model isn't just memorizing
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_data.tofile("train.bin")
val_data.tofile("val.bin")

# Save the vocab so train.py and generate.py can use the same encoding
with open("meta.pkl", "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
print("Saved train.bin, val.bin, meta.pkl -- ready for training.")
