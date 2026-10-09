"""Export instruction-tuned weights and a standalone CPU demo, without uploading.

    python export.py --checkpoint checkpoints/sft/best.pt --out checkpoints/release
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import torch
from safetensors.torch import save_model

from eval import load as load_checkpoint
from inference import FIELDS, TEMPLATE, digest, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/sft/best.pt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="checkpoints/release")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty. Use a new directory.")
    torch.set_num_threads(4)
    model, tok, cfg, step = load_checkpoint(args.checkpoint, args.data, "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "sft" not in checkpoint or checkpoint["sft"]["template"] != TEMPLATE:
        raise ValueError("Export requires an instruction-tuned checkpoint with the story template.")
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out / "model.safetensors"))
    shutil.copyfile(Path(args.data) / "tokenizer.model", out / "tokenizer.model")
    architecture = {key: cfg[key] for key in ("n_layer", "dim", "n_head", "ctx", "vocab")}
    architecture["pos"] = cfg.get("pos", "rope")
    root = Path(__file__).resolve().parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    config = dict(format_version=1, model=architecture, template=TEMPLATE, fields=FIELDS,
                  special_tokens=tok.special, dtype="float32", step=step,
                  source_checkpoint_sha256=digest(args.checkpoint), source_commit=commit,
                  training_fingerprint=checkpoint["fingerprint"],
                  instruction_revision=checkpoint["sft"]["revision"],
                  sha256={name: digest(out / name) for name in ("model.safetensors", "tokenizer.model")})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("model.py", "tokenizer.py", "inference.py", "app.py", "requirements.txt", "LICENSE"):
        shutil.copyfile(root / name, out / name)
    params = sum(p.numel() for p in model.parameters())
    run = checkpoint.get("wandb_id")
    card = f"""---
license: mit
language:
- en
datasets:
- roneneldan/TinyStories
- roneneldan/TinyStoriesInstruct
pipeline_tag: text-generation
tags:
- pytorch
- safetensors
- custom-code
---

# Sprout story model

A decoder-only model trained from scratch on TinyStoriesV2-GPT4 and instruction tuned on TinyStoriesInstruct. It has {params:,} parameters, {cfg['n_layer']} layers, width {cfg['dim']}, {cfg['n_head']} attention heads, a {cfg['ctx']}-token context, {architecture['pos']} positions, RMSNorm, SwiGLU, tied embeddings, and a {cfg['vocab']}-token byte-level BPE vocabulary.

[Try the live browser demo](https://huggingface.co/spaces/GeorgeCodesSometimes/sprout). The static demo runs float32 ONNX weights on your device; prompts stay local.

This bundle contains float32 inference weights at instruction step {step:,}. It excludes optimizer state, training data, credentials, and training dependencies. The tied embedding/output tensor is stored once and restored with `safetensors.torch.load_model`. `config.json` records weight and tokenizer SHA-256 checksums, source checkpoint SHA-256, source code commit, and the original training fingerprint.

## Run locally

Download all files from this model repository into an empty directory, then run:

```bash
python -m pip install -r requirements.txt
python app.py
```

Use the included loading module directly:

```python
from inference import load, stream

model, tok, config = load(".")
for text in stream(model, tok, {{"Summary": "A little fox learns to be patient."}}):
    print(text, end="\\r", flush=True)
```

`load` also accepts a Hub model ID and an optional `revision` commit. The included code is the loading contract; this is not a Transformers `AutoModel` repository. Loading does not require any training corpus files. The demo uses the exact training template, keeps the full instruction in context, and supports streamed output, stopping, temperature, and top-k.

## Training and evaluation

The source checkpoint's instruction validation loss is {checkpoint['best_val']:.6f}. The [training run](https://wandb.ai/george-nasseem/sprout/runs/{run}) records configuration and losses. See the [project README](https://github.com/gnasseem/sprout#instruction-tuning) for the fixed behavioral evaluation, comparison with the pretrained model, and base-validation regression. Low validation loss does not establish reliable constraint following.

Pretraining uses the story-level seeded split in the project, with 537,686,596 training tokens and 5,402,089 validation tokens. Instruction preparation excludes malformed and duplicate stories, held-out story overlap, and conversations beyond the context limit. TinyStoriesInstruct is pinned to `{config['instruction_revision']}`. These are synthetic English stories generated by GPT models, not a general conversational corpus.

## Limitations and intended use

An educational story-generation experiment. It often misses requested words and sentences, repeats itself, and loses narrative details. Instruction tuning regresses base-story validation loss. Long instructions leave less room for the story. It has no general chat or factual-answering capability, no content filter, and no suitability evaluation for children. Outputs can contain biases or disturbing story content from its synthetic source corpus.

## License and provenance

Code and exported weights use the included MIT license. The upstream [TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories) lists CDLA-Sharing-1.0; [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) is the separate instruction corpus. No source stories are redistributed in this bundle. The dataset creators' [paper](https://arxiv.org/abs/2305.07759) describes TinyStories.
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    (out / "SPACE.md").write_text("""---
title: Sprout Story Workshop
sdk: gradio
sdk_version: 6.30.0
python_version: '3.11'
app_file: app.py
pinned: false
license: mit
---

# Sprout Story Workshop

An experimental small story model. It can miss constraints, repeat itself, and lose story details.
""")
    restored, restored_tok, _ = load(out)
    prefix = tok.encode(TEMPLATE.format(instruction="Write a short story.\nSummary: A fox finds a seed."))
    x = torch.tensor([prefix])
    before = model.generate(x, 64, temperature=0)
    after = restored.generate(x, 64, temperature=0)
    assert torch.equal(before, after), "export changed greedy generation"
    assert model.head.weight is model.tok_emb.weight
    assert restored.head.weight is restored.tok_emb.weight
    assert restored_tok.decode(after[0].tolist()) == tok.decode(before[0].tolist())
    print(f"Exported {params:,} parameters; 64 greedy tokens match; tied weights and tokenizer match.")
    print(f"Bundle: {out.resolve()}; weights: {(out / 'model.safetensors').stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
