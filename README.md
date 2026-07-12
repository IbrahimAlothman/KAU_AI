# MiniGPT — Your Own Local Model, ChatGPT-Style

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

## Deploying outside your Mac (Railway + Netlify)

This matches how you usually deploy: **Railway** runs the backend
(needs real compute for PyTorch), **Netlify** hosts the static chat page.

### 1. Train first, then push everything to GitHub
Run `prepare_data.py` and `train.py` on your Mac first — you need
`checkpoint.pt` and `meta.pkl` to exist before deploying, since those
ARE your trained model. Then push the whole `mini_llm` folder
(including `checkpoint.pt` and `meta.pkl`) to a GitHub repo.

> `checkpoint.pt` will be a few MB — fine for a normal GitHub repo at
> this small model size.

### 2. Deploy the backend to Railway
1. Go to railway.app → New Project → Deploy from GitHub repo
2. Select your repo — Railway auto-detects Python and uses the
   `Procfile` we created to know how to start it
3. Once deployed, Railway gives you a public URL, something like
   `https://mini-llm-production.up.railway.app`
4. Test it works: visit `https://your-railway-url/api/health` in a
   browser — you should see `{"status": "ok", ...}`

### 3. Point the frontend at your deployed backend
In `index.html`, find this line near the bottom:
```js
const API_BASE = "http://localhost:5001";
```
Replace it with your Railway URL:
```js
const API_BASE = "https://mini-llm-production.up.railway.app";
```

### 4. Deploy the frontend to Netlify
Easiest way: go to netlify.com → drag and drop the `mini_llm` folder
(or just `index.html`) onto the deploy area. Netlify gives you a live
URL instantly — that's your public demo link.

### A note on security
`server.py` currently allows requests from any website (`CORS(app)`
with no restriction) so it's easy to test. Before sharing the link
widely, it's worth restricting this to only your Netlify domain —
happy to tighten that up when you're ready to share it more broadly.

## Taking this further

- Swap `prepare_data.py`'s source text for your own dataset (Arabic
  corpus, agency content, etc.) to make it "learn" something else
- Increase `embed_size`, `num_layers`, or `max_iters` in `train.py`
  for a stronger (but slower to train) model — better done on the
  university's GPUs rather than your Mac
- To make this reachable outside your Mac (a real public demo):
  deploy `server.py` to Railway (or similar) and `index.html` to
  Netlify, then update `API_URL` in `index.html` to point at the
  deployed server's address instead of localhost
