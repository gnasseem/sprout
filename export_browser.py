"""Export a cached float32 ONNX model for the static browser demo.

    pip install onnx onnxruntime
    python export_browser.py --model checkpoints/release --out checkpoints/browser
"""

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort
import torch

from inference import digest, load, prompt
from model import rope


class BrowserModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, cache):
        model = self.model
        past = cache.shape[4]
        length = tokens.shape[1]
        positions = past + torch.arange(length)
        x = model.tok_emb(tokens)
        if model.pos_emb is not None:
            x = x + model.pos_emb(positions)
        cos, sin = model.cos[positions], model.sin[positions]
        mask = torch.arange(past + length)[None, :] <= positions[:, None]
        updated = []
        for i, block in enumerate(model.blocks):
            attention = block.attn
            y = block.attn_norm(x)
            heads = attention.n_head
            q = attention.wq(y).reshape(1, length, heads, -1).transpose(1, 2)
            k = attention.wk(y).reshape(1, length, heads, -1).transpose(1, 2)
            v = attention.wv(y).reshape(1, length, heads, -1).transpose(1, 2)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            k = torch.cat((cache[i, 0], k), dim=2)
            v = torch.cat((cache[i, 1], v), dim=2)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(model.tok_emb.embedding_dim // heads)
            scores = scores.masked_fill(~mask, float("-inf"))
            y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, length, -1)
            x = x + attention.wo(y)
            x = x + block.mlp(block.mlp_norm(x))
            updated.append(torch.stack((k, v)))
        return model.head(model.norm(x[:, -1])), torch.stack(updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/release")
    parser.add_argument("--out", default="checkpoints/browser")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty. Use a new directory.")
    torch.set_num_threads(4)
    model, tok, config = load(args.model)
    wrapper = BrowserModel(model).eval()
    shape = config["model"]
    empty = torch.empty(shape["n_layer"], 2, 1, shape["n_head"], 0,
                        shape["dim"] // shape["n_head"])
    tokens = torch.tensor([prompt(tok, {"Summary": "A fox finds a seed."})])
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, (tokens, empty), str(out / "model.onnx"),
                          input_names=["tokens", "cache"], output_names=["logits", "present"],
                          dynamic_axes={"tokens": {1: "length"}, "cache": {4: "past"},
                                        "present": {4: "total"}},
                          opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(out / "model.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    cache = empty.numpy()
    reference_cache = None
    with torch.no_grad():
        for step in range(64):
            logits, cache = session.run(None, {"tokens": tokens.numpy(), "cache": cache})
            reference, reference_cache = model.forward_cached(tokens, reference_cache)
            np.testing.assert_allclose(logits, reference[:, -1].numpy(), atol=1e-4, rtol=1e-4)
            assert logits.argmax() == reference[:, -1].argmax().item(), step
            tokens = torch.tensor([[int(logits.argmax())]])
    shutil.copyfile(Path(args.model) / "tokenizer.model", out / "tokenizer.model")
    config["browser_sha256"] = digest(out / "model.onnx")
    config["browser_bytes"] = (out / "model.onnx").stat().st_size
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"64 cached greedy tokens and logits match PyTorch; ONNX: {config['browser_bytes']:,} bytes.")


if __name__ == "__main__":
    main()
