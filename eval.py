"""Evaluate a checkpoint and print every seeded story continuation, unedited.

    python eval.py checkpoints/main/ckpt.pt
    python eval.py checkpoints/vocab8192/ckpt.pt --data data/vocab8192
"""

import argparse
import hashlib
import json

import torch

import tokenizer
from model import Sprout
from train import DEVICE, SEED, autocast, fingerprint, split, token_bytes, val_metrics

PROMPTS = (
    "Once upon a time",
    "Lily found a small red box under the old tree.",
    "Tom wanted to fly, but he did not have wings.",
    "The little rabbit was afraid of the dark.",
    "One rainy day, Anna heard a knock at the door.",
    "Ben and Mia were best friends. Today, they had an argument.",
    "A tiny dragon could only blow bubbles instead of fire.",
    "Sara planted a seed and waited for it to grow.",
    "Max lost his favorite blue hat at the park.",
    "The puppy saw its own face in the mirror.",
    "Lucy promised her mother that she would not open the door.",
    "Jack had three apples. He gave one apple to his sister.",
    "The snowman wanted to see what summer was like.",
    "Every morning, the little bird sang the same song. But today",
    "Nora found a key, but she did not know what it could open.",
    "The toy boat sank when Sam put a heavy stone inside it.",
    "A fox and a duck wanted to cross the river together.",
    "Emma was sad because nobody had remembered her birthday.",
    "Leo wore a red coat. His sister Ada wore a yellow coat. They traded coats.",
    "Rose put her marble in a green cup. While Rose was outside, Finn moved the "
    "marble into a blue cup. Rose came back and looked for her marble.",
)

def load(path, data, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["fingerprint"] == fingerprint(data), "checkpoint and data do not match"
    cfg = ckpt["config"]
    tok = tokenizer.load(f"{data}/tokenizer.model")
    assert len(tok.vocab) == cfg["vocab"]
    model = Sprout(cfg["n_layer"], cfg["dim"], cfg["n_head"], cfg["ctx"],
                   cfg["vocab"], cfg.get("pos", "rope"))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), tok, cfg, ckpt["step"]

@torch.inference_mode()
def continuation(model, tok, prompt, args, seed):
    torch.manual_seed(seed)
    if args.device == "mps":
        torch.mps.manual_seed(seed)
    eot = tok.special["<|endoftext|>"]
    prefix = [eot] + tok.encode(prompt)
    assert len(prefix) <= model.ctx, "prompt exceeds the model context"
    idx = torch.tensor([prefix], device=args.device)
    with autocast(args.device):
        ids = model.generate(idx, args.max_tokens, args.temperature, eot, args.top_k)
    ids = ids[0, len(prefix):].tolist()
    ended = eot in ids
    if ended:
        ids = ids[:ids.index(eot)]
    return tok.decode(ids), ids, ended

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--data", default="data")
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default=DEVICE)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()
    assert args.micro_batch > 0 and args.max_tokens > 0
    assert args.temperature > 0 and args.top_k > 0
    model, tok, cfg, step = load(args.checkpoint, args.data, args.device)
    assert args.top_k <= cfg["vocab"]
    args.ctx = cfg["ctx"]
    with open(args.checkpoint, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    print(json.dumps(dict(checkpoint=args.checkpoint, sha256=digest, step=step,
                          config=cfg, fingerprint=fingerprint(args.data),
                          torch=torch.__version__, device=args.device,
                          dtype="bfloat16" if args.device == "cuda" else "float32",
                          micro_batch=args.micro_batch)), flush=True)
    data = split(args.data, "val")
    nbytes = token_bytes(tok)
    loss, bpb = val_metrics(model, data, nbytes, args)
    print(json.dumps(dict(val_loss=loss, val_bpb=bpb, targets=len(data) - 1,
                          bytes=int(nbytes[data[1:]].sum()), context=args.ctx)), flush=True)
    for i, prompt in enumerate(PROMPTS):
        seed = args.seed + i
        text, ids, ended = continuation(model, tok, prompt, args, seed)
        if i == 0:
            repeated = continuation(model, tok, prompt, args, seed)
            assert (text, ids, ended) == repeated, "seeded sampling did not reproduce"
        print(json.dumps(dict(prompt_id=i + 1, prompt=prompt, continuation=text,
                              token_ids=ids, seed=seed, temperature=args.temperature,
                              top_k=args.top_k, max_tokens=args.max_tokens,
                              generated_tokens=len(ids), ended=ended), ensure_ascii=False),
              flush=True)

if __name__ == "__main__":
    main()
