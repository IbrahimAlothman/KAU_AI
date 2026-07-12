"""
STEP 3: Train the model.

This is where learning actually happens. In simple terms, the loop is:
  1. Grab a random chunk of text from the training data
  2. Ask the model to predict the next character at every position
  3. Compare its predictions to the real next characters (this is the "loss" -- 
     a number that's high when the model is wrong, low when it's right)
  4. Adjust the model's internal numbers slightly to reduce that loss
  5. Repeat thousands of times

Run:  python train.py
On an M1/M2/M3 Mac this takes roughly 15-30 minutes for a good demo result.
"""

import torch
import pickle
import time
from model import MiniGPT

# ---- Settings (small on purpose, so it trains fast on a Mac) ----
block_size = 128      # how many characters of context the model looks at
batch_size = 32        # how many chunks we train on at once
embed_size = 128       # size of the model's internal representation
num_heads = 4
num_layers = 4
dropout = 0.1
learning_rate = 3e-4
max_iters = 3000
eval_interval = 300
# -------------------------------------------------------------------

# Use Apple Silicon GPU if available, otherwise CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Training on device: {device}")

# Load the data we prepared in step 1
with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]

import numpy as np
train_data = torch.from_numpy(np.fromfile("train.bin", dtype=np.uint16).astype(np.int64))
val_data = torch.from_numpy(np.fromfile("val.bin", dtype=np.uint16).astype(np.int64))

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(50)
        for k in range(50):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

model = MiniGPT(vocab_size, embed_size, num_heads, num_layers, block_size, dropout).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start = time.time()
for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - start
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}  ({elapsed:.0f}s elapsed)")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the trained model so we can chat with it later
torch.save({
    "model_state": model.state_dict(),
    "vocab_size": vocab_size,
    "embed_size": embed_size,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "block_size": block_size,
    "dropout": dropout,
}, "checkpoint.pt")

print("Training done. Saved checkpoint.pt")
