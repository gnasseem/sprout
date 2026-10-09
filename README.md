# Sprout

A **29.4 million parameter language model trained from scratch** to write simple English stories. Eight transformer layers, a byte-level BPE tokenizer, and a complete training-to-inference pipeline in plain PyTorch.

**[Write a story in your browser](https://huggingface.co/spaces/GeorgeCodesSometimes/sprout)** · [Download the weights](https://huggingface.co/GeorgeCodesSometimes/sprout-instruct) · [Training and reproduction](docs/reproduction.md)

Sprout is a finished educational experiment: pretraining, two controlled comparisons, instruction tuning, evaluation, and public inference weights are complete. It produces readable short stories and shows how a small language model is built. It also repeats itself, loses narrative details, and often misses instructions. It is useful as a learning project, not a general chat assistant or a reliable writing product.

## Try it

The [live demo](https://huggingface.co/spaces/GeorgeCodesSometimes/sprout) runs the actual instruction-tuned model on your device using ONNX Runtime Web. First use downloads about **126 MB** of float32 weights plus the runtime; subsequent visits can reuse the browser cache. Your story prompts stay on your device. No account, API key, Python installation, or paid inference server is needed.

Enter a story idea, optionally request words or a sentence, and click **Load model & write**. Stories stream as they are generated. You can stop, copy the text, or change temperature and length. CPU generation can be slow on phones; the Python demo below is another way to run it.

## Run locally

Use Python 3.11 or later:

```bash
git clone https://github.com/gnasseem/sprout.git
cd sprout
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open [localhost:7860](http://127.0.0.1:7860). The app downloads the public inference bundle automatically when no local bundle exists. On Windows, activate with `.venv\Scripts\activate`.

To select a model directory or pin a Hub revision:

```bash
python app.py --model GeorgeCodesSometimes/sprout-instruct --revision 70a25ae2b0e456b07813895d98e1a29f8d3d449e
```

For a script without the UI:

```python
from inference import load, stream

model, tok, config = load("GeorgeCodesSometimes/sprout-instruct")
for story in stream(model, tok, {"Summary": "A fox finds a seed."}):
    print(story, end="\r", flush=True)
```

`stream` yields the complete story so far. This is a custom PyTorch model; the included `inference.py` is its loading interface, rather than Transformers `AutoModel`. The bundle verifies weight and tokenizer checksums, restores tied embeddings, and needs no training corpus.

## Results

The main pretrained model saw **1.573 billion tokens** and achieved **0.391208 bits per byte** on complete held-out validation. All pretraining rows below use the same complete scoring policy, including the final partial window, evaluated in MPS float32. Compare vocabularies using bits per byte, since their tokens differ.

| Run | Training tokens | Validation loss, nats/token | Bits per byte | Record |
| --- | ---: | ---: | ---: | --- |
| Main model | 1,572,864,000 | 1.091364 | 0.391208 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/y5ye7t4k) |
| One-pass RoPE baseline | 537,657,344 | 1.180857 | 0.423287 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/i0wis53q) |
| One-pass larger vocabulary | 520,880,128 | 1.202189 | 0.417522 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/yf7r4ema) |
| One-pass learned positions | 537,657,344 | 1.196941 | 0.429052 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/ul3wfup5) |

The controlled runs use approximately one corpus-equivalent text budget, sampled with replacement. Doubling the vocabulary to 8,192 improves bits per byte by **1.36%**, with **7.14% more parameters** and fewer updates. Learned positions worsen bits per byte by **1.36%** compared with RoPE. These are single-seed comparisons, not evidence of universal architectural superiority; the longer main run is outside the controlled comparison.

![Pretraining, comparison runs, and instruction-tuning loss curves](docs/curves.svg)

Instruction tuning completed one shuffled epoch over **2,362,399 conversations**, with assistant-only loss. Its validation loss fell from **1.419628 to 0.956147**. A fixed set of 20 official held-out prompts, evaluated on matching CUDA bf16 settings, shows what that improvement does and does not buy:

| Behavior | Before tuning | After tuning |
| --- | ---: | ---: |
| All requested words included | 0/20 | 6/20 |
| Exact requested sentence included | 1/20 | 4/20 |
| Stopped with the assistant end token | 0/20 | 19/20 |
| Base-story validation loss, nats/token | 1.091467 | 1.220414 |

Stopping improves substantially, while base-story validation loss rises **11.81%** and most literal constraints still fail. These small suites measure words, sentences, and stopping; story quality and summary fidelity were not graded. [Unedited evaluation outputs](https://huggingface.co/GeorgeCodesSometimes/sprout-instruct/tree/main/evaluation) retain both successes and failures. The [detailed documentation](docs/reproduction.md#instruction-tuning) describes prompt selection, settings, and evaluation precision.

For example, one tuned output includes the requested sentence and two requested words, but substitutes “tried” for the required whole word “try”:

> Once upon a time, there was a little furry cat. The cat liked to look out the window and watch the birds fly by. One day, the cat wanted to play with the birds, but they were too high up in the sky.

## What is inside

| File | Purpose |
| --- | --- |
| `data.py` | Pinned TinyStories download and deterministic story-level split |
| `tokenizer.py` | Unicode-aware byte BPE and corpus tokenization |
| `model.py` | Transformer, cached generation, and inline correctness checks |
| `train.py` | Pretraining, full validation, checkpointing, and exact resume |
| `sft.py` | Instruction preparation, masked training, and behavioral evaluation |
| `eval.py` | Complete validation and seeded story-generation suite |
| `inference.py` | Verified portable loading and UTF-8 text streaming |
| `export.py` | Safetensors bundle and standalone Python demo export |
| `export_browser.py` | Cached float32 ONNX export, checked against PyTorch |
| `app.py` | Streaming Gradio story workshop |
| `docs/` | Static browser demo, loss curves, and reproduction details |

The architecture has **8 layers**, **512 hidden dimensions**, **8 attention heads**, a **4,096-token vocabulary**, and a **512-token context**. It uses RoPE, RMSNorm, SwiGLU, no projection biases, and tied token embeddings/output weights. Inference uses a KV cache. [Architecture counts and data provenance](docs/reproduction.md#architecture) explain the choices and exact parameter totals.

Generated corpora, checkpoints, and local environments stay in ignored `data/`, `checkpoints/`, and `.venv/` directories. The public inference bundle is about 112 MiB; training checkpoints are larger because they include optimizer and RNG state. Keep original checkpoints and evaluation logs if you want to resume or inspect the runs.

## Reproduce the training

Prepare the data and tokenizer, then run the existing inline checks and main training run:

```bash
python data.py
python tokenizer.py
python model.py --device cpu
python train.py --check
python train.py --out checkpoints/main --steps 12000 --no-wandb
python sft.py --prepare
python sft.py --check
python sft.py --checkpoint checkpoints/main/ckpt.pt --out checkpoints/sft --no-wandb
python eval.py checkpoints/main/ckpt.pt
python sft.py --checkpoint checkpoints/sft/best.pt --evaluate
```

Full training is intended for a CUDA GPU. Install a CUDA-compatible PyTorch build; install `wandb` separately if you want experiment tracking. The five logged runs total **4.81 hours**, including evaluation overhead. Applying the main run’s recorded A100 rate to all runs estimates about **$1.95** of compute; this is not a verified total bill. Original training used PyTorch `2.11.0+cu128`; later versions and devices can change sampled outputs.

See [training and reproduction](docs/reproduction.md#reproduce) for both comparison arms, exact settings, pinned dataset revisions, held-out metrics, device differences, and resume commands. The private corpus/checkpoint repositories are optional shortcuts; public upstream data and this source are sufficient to train from scratch.

## Export and host

Export a fresh Python bundle and run it:

```bash
python export.py --checkpoint checkpoints/sft/best.pt --out checkpoints/release
python app.py --model checkpoints/release
```

Exports require a new or empty output directory. For browser weights, install the optional export dependencies and export the Python bundle:

```bash
python -m pip install onnx==1.23.2 onnxruntime==1.31.0
python export_browser.py --model checkpoints/release --out checkpoints/browser
```

The browser export checks logits and 64 greedy tokens against PyTorch. It preserves float32 weights, uses dynamic cached attention, and checksums the model and tokenizer. `docs/worker.js` pins the hosted ONNX bundle to a specific Hub commit and ONNX Runtime Web to `1.30.0`. Browser and Python random samples can differ; greedy output is checked during export.

The demo is hosted as a [free static Hugging Face Space](https://huggingface.co/docs/hub/spaces-sdks-static). To also publish it with GitHub Pages, push this commit, then choose **Settings → Pages → Deploy from a branch → main → /docs** in the GitHub repository. Pages serves the same static files; model inference happens in the browser. To preview locally:

```bash
python -m http.server 8000 --directory docs
```

Open [localhost:8000](http://localhost:8000); ES modules and workers require HTTP rather than opening `index.html` as a file.

## Limits and license

Sprout is trained on synthetic English TinyStories and TinyStoriesInstruct data. It has no general factual-answering or chat capability, no content filter, and no suitability evaluation for children. The short context restricts instructions and story length. A finished training pipeline does not imply reliable instruction following, long-range consistency, or performance outside this domain.

Code and exported weights use the [MIT license](LICENSE). The upstream [TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories) lists CDLA-Sharing-1.0; no training corpus is redistributed here. See the [TinyStories paper](https://arxiv.org/abs/2305.07759) and [provenance documentation](docs/reproduction.md#data-and-tokenizer).
