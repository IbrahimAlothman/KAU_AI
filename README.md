# KAU AI — Your Own Local Model

A small transformer language model, trained from scratch (random weights)
on your Mac, served through a simple API, with a ChatGPT-style web
interface on top. Nothing here is a pretrained model you downloaded —
every number in it is learned from the training run you do yourself.

## What each file does

- `prepare_data.py` — downloads Shakespeare text, converts it to numbers
- `model.py` — the transformer architecture itself (the "brain")
- `train.py` — trains the model from scratch on your Mac
- `server.py` — loads the trained model and exposes a chat API
- `index.html` — the ChatGPT-style web interface

## Setup (do this once)

1. Open Terminal, navigate to this folder:
   ```
   cd path/to/mini_llm
   ```

2. Create a virtual environment (keeps this project's packages separate):
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run it — 3 steps

**Step 1 — Prepare the data** (downloads and processes the text, ~10 seconds)
```
python prepare_data.py
```

**Step 2 — Train the model** (this is the actual "building our own model" part)
```
python train.py
```
This takes roughly 15–30 minutes on an M1/M2/M3 Mac. You'll see the loss
number dropping every 300 steps — that's the model getting better at
predicting Shakespeare's next character. When it's done, it saves
`checkpoint.pt` — that file IS your trained model.

**Step 3 — Start the server**
```
python server.py
```
Leave this running in the terminal — it's now serving your model at
`http://localhost:5001`.

**Step 4 — Open the chat interface**
Just double-click `index.html`, or open it in your browser. Type a
message and watch your own trained model respond.

## What to expect

Because this is a small model trained only on Shakespeare, it won't
answer questions or hold a real conversation — it will continue
whatever you type in a Shakespeare-like style. That's the honest,
expected result of a demo at this scale, and it's still proof the
model learned real patterns from data, entirely on your machine.


