"""
STEP 4: Serve the model as a web API.

This loads the trained checkpoint and exposes a /api/chat endpoint.
The frontend (index.html) will send your message here, and this will
respond with the model's generated continuation.

Run:  python server.py
Then open index.html in your browser (or serve it too -- see README).
"""

import os
import time
import torch
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import MiniGPT

device = "mps" if torch.backends.mps.is_available() else "cpu"

with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

checkpoint = torch.load("checkpoint.pt", map_location=device)
model = MiniGPT(
    vocab_size=checkpoint["vocab_size"],
    embed_size=checkpoint["embed_size"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    block_size=checkpoint["block_size"],
    dropout=checkpoint["dropout"],
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

block_size = checkpoint["block_size"]

def encode(s):
    # any character not seen during training gets skipped safely
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return "".join([itos[i] for i in l])

app = Flask(__name__)
# Only allow requests from your live frontend (plus localhost for testing on your Mac)
CORS(app, origins=[
    "https://monumental-raindrop-3770ab.netlify.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
])

@app.route("/api/chat", methods=["POST"])
def chat():
    start = time.time()
    print(f"[chat] request received at {start:.0f}", flush=True)
    data = request.json
    user_message = data.get("message", "")
    # Cap generation length -- this is a small CPU-only model, so keep responses
    # short enough to finish comfortably within the request timeout.
    max_new_tokens = min(int(data.get("max_tokens", 40)), 40)

    # The model was trained exclusively on "Question: ...\nAnswer: ..." pairs --
    # it only knows how to respond sensibly when the input matches that exact
    # shape. Wrapping the user's message the same way is what actually makes
    # this behave like a Q&A bot instead of free-associating from raw text.
    prompt = f"Question: {user_message}\nAnswer:"

    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    if context.shape[1] == 0:
        context = torch.tensor([[0]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=float(data.get("temperature", 0.7)),
            top_k=int(data.get("top_k", 10)),
        )

    full_text = decode(out[0].tolist())
    # Only keep what comes after "Answer:". Training pairs are separated by a
    # blank line (double newline), so cut there rather than matching the word
    # "Question" -- generation can get cut off mid-word right at that point.
    reply = full_text[len(prompt):]
    reply = reply.split("\n\n")[0]
    reply = reply.strip()

    print(f"[chat] {max_new_tokens} tokens generated in {time.time() - start:.1f}s -- reply: {reply!r}", flush=True)
    return jsonify({"reply": reply})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})

if __name__ == "__main__":
    # Railway (and most hosts) assign a port via the PORT environment variable --
    # falls back to 5001 for local runs on your Mac.
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
