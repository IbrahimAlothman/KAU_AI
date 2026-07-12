"""
STEP 4: Serve the model as a web API.

This loads the trained checkpoint and exposes a /api/chat endpoint.
The frontend (index.html) will send your message here, and this will
respond with the model's generated continuation.

Run:  python server.py
Then open index.html in your browser (or serve it too -- see README).
"""

import os
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
CORS(app)  # allows your frontend (even on a different port) to call this API

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    max_new_tokens = int(data.get("max_tokens", 200))

    context = torch.tensor([encode(user_message)], dtype=torch.long, device=device)
    if context.shape[1] == 0:
        context = torch.tensor([[0]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(context, max_new_tokens=max_new_tokens)

    full_text = decode(out[0].tolist())
    # strip the echoed input so we only return the newly generated part
    reply = full_text[len(user_message):]

    return jsonify({"reply": reply})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})

if __name__ == "__main__":
    # Railway (and most hosts) assign a port via the PORT environment variable --
    # falls back to 5001 for local runs on your Mac.
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
