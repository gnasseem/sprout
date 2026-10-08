"""Pretraining loop: AdamW, warmup into cosine decay, bf16 autocast, gradient
accumulation, resumable checkpoints. Batches are random windows of the .bin files.

    python train.py            # train with the constants below
    python train.py --check    # the checks that say the loop is correct
    python train.py --toy      # 2 layers, 128 dim: the whole path in minutes on a laptop
    python train.py --data data/vocab8192 --pos learned ...   # ablation arms
"""

import argparse
import contextlib
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import tokenizer
from model import CTX, DIM, N_HEAD, N_LAYER, VOCAB_SIZE, Sprout

DATA = "data"
OUT = "checkpoints"

TOKENS_PER_STEP = 1 << 17  # 131,072 tokens per optimizer step
MICRO_BATCH = 64  # sequences per forward pass; the rest is accumulation
MAX_STEPS = 12_000  # 1.57B tokens, ~2.9 epochs of the 538M-token corpus
WARMUP = 300
PEAK_LR = 1e-3
FLOOR = 0.1  # cosine floor as a fraction of peak
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
EVAL_EVERY = 250
EVAL_BATCHES = 40
CKPT_EVERY = 2000  # about every 15 minutes on an A100
SEED = 1337
PROJECT = "sprout"
HUB = ""  # e.g. "user/sprout-checkpoints": every checkpoint is also pushed here
TOY = dict(layers=2, dim=128, heads=4, ctx=256, micro_batch=32, accum=1, out=f"{OUT}/toy",
           steps=1500, warmup=100, lr=1e-3)
PROMPT = "Once upon a time"

ACCUM = TOKENS_PER_STEP // (MICRO_BATCH * CTX)
assert MICRO_BATCH * CTX * ACCUM == TOKENS_PER_STEP

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def split(data, name):
    return np.memmap(f"{data}/{name}.bin", dtype=np.uint16, mode="r")

def batch(data, size, ctx, gen, device):
    """Random windows, sampled with replacement rather than by epoch permutation.
    Over 538M tokens the two look the same, and this way the generator state is
    the whole position in the data, which is what makes a resume exact."""
    i = torch.randint(len(data) - ctx, (size,), generator=gen)
    window = np.stack([data[j : j + ctx + 1] for j in i.tolist()])
    t = torch.from_numpy(window.astype(np.int64))
    x, y = t[:, :-1], t[:, 1:]
    if device == "cuda":
        x, y = x.pin_memory(), y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x.to(device), y.to(device)

def lr_at(step, steps, warmup, peak=PEAK_LR):
    """Linear warmup, then cosine down to FLOOR of peak. Once per optimizer
    step: per micro-batch it would decay ACCUM times too fast."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, steps - warmup))
    return peak * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * t)))

def optimizer(model, device):
    """AdamW, decaying the matrices but not the RMSNorm gains: a gain sets the
    scale of a whole activation vector, so shrinking it mutes the layer."""
    # The tied embedding does get decayed. Embeddings are usually spared
    # because a row only gets a gradient when its token appears, but tying
    # makes it the output head, where every row moves every step.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    plain = [p for p in model.parameters() if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": plain, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=PEAK_LR, betas=BETAS, fused=(device == "cuda")
    )

def autocast(device):
    # No GradScaler: bf16 keeps fp32's exponent, so gradients cannot underflow
    # the way fp16's can. What it gives up is mantissa bits, which no amount of
    # scaling helps.
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

def fingerprint(data):
    """What the tokens were made of. Stored in every checkpoint so a rerun of
    the tokenizer cannot silently invalidate a resume or an eval."""
    with open(f"{data}/tokenizer.model", "rb") as f:
        merges = hashlib.blake2b(f.read(), digest_size=8).hexdigest()
    return {
        "tokenizer": merges,
        "train_tokens": os.path.getsize(f"{data}/train.bin") // 2,
        "val_tokens": os.path.getsize(f"{data}/val.bin") // 2,
    }

def token_bytes(tok):
    """Bytes of text behind each token id; 0 for the specials, which are not text."""
    n = np.array([len(tok.vocab[i]) for i in range(len(tok.vocab))], dtype=np.int64)
    n[list(tok.special.values())] = 0
    return n

def config(args):
    """Everything a resume has to agree on."""
    return dict(
        steps=args.steps, warmup=args.warmup, ctx=args.ctx,
        micro_batch=args.micro_batch, accum=args.accum,
        tokens_per_step=args.micro_batch * args.ctx * args.accum,
        peak_lr=args.lr, floor=FLOOR, weight_decay=WEIGHT_DECAY,
        betas=BETAS, grad_clip=GRAD_CLIP, seed=SEED,
        n_layer=args.layers, dim=args.dim, n_head=args.heads, vocab=args.vocab,
        pos=args.pos,
    )

@torch.no_grad()
def val_loss(model, data, args, batches=EVAL_BATCHES):
    was_training = model.training
    model.eval()
    # A fresh generator every time, so every eval sees the same windows and the
    # curve moves because the model moved.
    gen = torch.Generator().manual_seed(SEED + 1)
    total = 0.0
    for _ in range(batches):
        x, y = batch(data, args.micro_batch, args.ctx, gen, args.device)
        with autocast(args.device):
            total += model(x, y)[1].item()
    model.train(was_training)
    return total / batches

@torch.no_grad()
def val_metrics(model, data, nbytes, args):
    """Cross-entropy and bits per byte in consecutive windows, including the tail.
    Every token after the first is a target; special tokens count as zero bytes."""
    assert len(data) > 1
    was_training = model.training
    model.eval()
    ntokens = len(data) - 1
    nats = 0.0
    start = 0
    try:
        while start < ntokens:
            width = min(args.ctx, ntokens - start)
            count = min(args.micro_batch, (ntokens - start) // width)
            t = torch.from_numpy(data[start:start + count * width + 1].astype(np.int64))
            x, y = t[:-1].view(count, width), t[1:].view(count, width)
            with autocast(args.device):
                loss = model(x.to(args.device), y.to(args.device))[1]
            nats += loss.item() * y.numel()
            start += y.numel()
    finally:
        model.train(was_training)
    ntext = int(nbytes[data[1:]].sum())
    assert ntext > 0
    return nats / ntokens, nats / math.log(2) / ntext

def val_bpb(model, data, nbytes, args):
    return val_metrics(model, data, nbytes, args)[1]

def sample(model, tok, device, n=150):
    """Continue PROMPT from a fresh story, sampling until the story ends."""
    eot = tok.special["<|endoftext|>"]
    idx = torch.tensor([[eot] + tok.encode(PROMPT)], device=device)
    was_training = model.training
    model.eval()
    out = model.generate(idx, n, temperature=0.8, stop=eot)[0, 1:].tolist()
    model.train(was_training)
    return tok.decode(out[: out.index(eot)] if eot in out else out)

def save(path, model, opt, gen, step, args, fp):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "config": config(args),
        "fingerprint": fp,
        "data_rng": gen.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(ckpt, path + ".tmp")
    os.replace(path + ".tmp", path)  # a crash mid-write leaves the old one intact
    if args.hub:
        # The instance disk dies with the instance; the Hub copy is the real one.
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=path, path_in_repo=os.path.relpath(path, OUT),
                            repo_id=args.hub, commit_message=f"step {step}")

def resume(path, model, opt, gen, fp, args):
    """Restore weights, Adam state and both RNGs; return the step to go on from."""
    if not os.path.exists(path) and args.hub:
        from huggingface_hub import hf_hub_download
        hf_hub_download(args.hub, os.path.relpath(path, OUT), local_dir=OUT)
    # Onto the CPU whatever the device: load_state_dict moves tensors to where
    # the parameters already live, and the RNG states have to stay CPU bytes.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["fingerprint"] == fp, (ckpt["fingerprint"], fp)
    assert ckpt["config"] == config(args), (ckpt["config"], config(args))
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    gen.set_state(ckpt["data_rng"])
    torch.set_rng_state(ckpt["torch_rng"])
    if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return ckpt["step"]

def run(args):
    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/ckpt.pt"
    fp = fingerprint(args.data)
    tok = tokenizer.load(f"{args.data}/tokenizer.model")
    args.vocab = len(tok.vocab)

    torch.manual_seed(SEED)
    model = Sprout(args.layers, args.dim, args.heads, args.ctx, args.vocab, args.pos)
    model = model.to(args.device)
    opt = optimizer(model, args.device)
    gen = torch.Generator().manual_seed(SEED)
    step = resume(path, model, opt, gen, fp, args) if args.resume else 0

    train_data, val_data = split(args.data, "train"), split(args.data, "val")
    nbytes = token_bytes(tok)
    bytes_per_token = nbytes[val_data].sum() / len(val_data)
    tokens = args.micro_batch * args.ctx * args.accum
    stop = args.stop_after or args.steps
    epochs = args.steps * tokens / len(train_data)
    print(f"{sum(p.numel() for p in model.parameters()):,} params on {args.device}, "
          f"{tokens:,} tokens/step x {args.steps:,} steps = {epochs:.2f} epochs")
    print(f"resuming at step {step}" if args.resume else f"fingerprint {fp}")

    wandb = None
    if not args.no_wandb:
        import wandb
        # WANDB_RUN_ID before a resume continues the same run instead of
        # opening a second one; WANDB_MODE=offline covers headless machines.
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        wandb.init(project=PROJECT, name=os.path.basename(args.out),
                   config=config(args) | fp | {"commit": commit}, resume="allow")

    losses = open(f"{args.out}/losses.txt", "a")
    vals = open(f"{args.out}/val.txt", "a")
    model.train()
    t0 = time.time()
    while step < stop:
        lr = lr_at(step, args.steps, args.warmup, args.lr)
        for group in opt.param_groups:
            group["lr"] = lr

        total = 0.0
        for _ in range(args.accum):
            x, y = batch(train_data, args.micro_batch, args.ctx, gen, args.device)
            with autocast(args.device):
                loss = model(x, y)[1]
            # Divided here so the accumulated gradient is the mean over the
            # whole step, not the sum of ACCUM micro-batch means.
            (loss / args.accum).backward()
            total += loss.item() / args.accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        dt = time.time() - t0
        t0 = time.time()
        print(f"step {step} {total:.8f}", file=losses, flush=True)
        print(f"step {step:6d}  loss {total:.4f}  lr {lr:.2e}  "
              f"grad {grad_norm:.3f}  {tokens / dt:,.0f} tok/s  {dt * 1e3:.0f}ms")
        if wandb:
            wandb.log({"loss": total, "lr": lr, "grad_norm": grad_norm.item(),
                       "tokens_per_sec": tokens / dt, "tokens": step * tokens}, step=step)

        if step % EVAL_EVERY == 0 or step == stop:
            val = val_loss(model, val_data, args)
            bpb = val / math.log(2) / bytes_per_token
            text = sample(model, tok, args.device)
            print(f"step {step} {val:.6f} {bpb:.6f}", file=vals, flush=True)
            print(f"step {step:6d}  val {val:.4f}  bpb {bpb:.4f}\n{text}\n")
            if wandb:
                wandb.log({"val_loss": val, "val_bpb": bpb,
                           "sample": wandb.Html(f"<pre>{text}</pre>")}, step=step)
        if step % CKPT_EVERY == 0 or step == stop:
            save(path, model, opt, gen, step, args, fp)

    if step == args.steps:
        bpb = val_bpb(model, val_data, nbytes, args)
        print(f"final val bpb over all {len(val_data):,} val tokens: {bpb:.4f}")
        print(f"final {bpb:.6f}", file=vals, flush=True)
        if wandb:
            wandb.summary["val_bpb_full"] = bpb

def check_schedule():
    """Warmup rises to the peak, cosine falls to the floor."""
    schedule = [lr_at(s, MAX_STEPS, WARMUP) for s in range(MAX_STEPS)]
    rising = all(b > a for a, b in zip(schedule, schedule[1:WARMUP]))
    falling = all(b <= a for a, b in zip(schedule[WARMUP - 1:], schedule[WARMUP:]))
    print(f"lr: start {schedule[0]:.2e}, peak {max(schedule):.2e} at step "
          f"{schedule.index(max(schedule)) + 1}, end {schedule[-1]:.2e} "
          f"(floor {PEAK_LR * FLOOR:.2e}), rising {rising}, then falling {falling}")
    assert rising and falling
    assert schedule[WARMUP - 1] == PEAK_LR == max(schedule)
    assert abs(schedule[-1] - PEAK_LR * FLOOR) < 1e-8

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(schedule)
    plt.xlabel("optimizer step")
    plt.ylabel("learning rate")
    plt.savefig(f"{OUT}/lr_schedule.png", dpi=120, bbox_inches="tight")
    print(f"wrote {OUT}/lr_schedule.png")

def check_eval(model, val_data, args):
    """Eval mode and no grad, and the model comes back the way it went in."""
    args.micro_batch, args.ctx = 8, 128
    model.train()
    val = val_loss(model, val_data, args, batches=4)
    print(f"val loss at init: {val:.4f} (ln V = {math.log(VOCAB_SIZE):.4f}), "
          f"training mode restored: {model.training}, "
          f"grads still empty: {all(p.grad is None for p in model.parameters())}")
    assert abs(val - math.log(VOCAB_SIZE)) < 0.15
    assert model.training and all(p.grad is None for p in model.parameters())

def check_accumulation(model, val_data):
    """One batch of 32 == four accumulated micro-batches of 8."""
    gen = torch.Generator().manual_seed(0)
    x, y = batch(val_data, 32, 128, gen, "cpu")
    whole = model(x, y)[1]
    whole.backward()
    full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)

    accumulated = 0.0
    for i in range(4):
        loss = model(x[i * 8:(i + 1) * 8], y[i * 8:(i + 1) * 8])[1]
        (loss / 4).backward()
        accumulated += loss.item() / 4
    scale = max(g.abs().max().item() for g in full)
    worst = max((a - p.grad).abs().max().item() for a, p in zip(full, model.parameters()))
    print(f"accumulation: loss {whole.item():.6f} vs {accumulated:.6f} "
          f"(diff {abs(whole.item() - accumulated):.2e}), "
          f"worst grad diff {worst:.2e} against grad scale {scale:.2e}")
    assert abs(whole.item() - accumulated) < 1e-5
    assert worst / scale < 1e-4
    model.zero_grad(set_to_none=True)

def check_resume():
    """100 steps in one process against 50 + checkpoint + fresh process + 50."""
    straight, interrupted = f"{OUT}/check-straight", f"{OUT}/check-interrupted"
    common = [sys.executable, __file__, "--device", "cpu", "--no-wandb",
              "--steps", "100", "--warmup", "10", "--ctx", "128",
              "--micro-batch", "4", "--accum", "2"]
    for out in (straight, interrupted):
        shutil.rmtree(out, ignore_errors=True)
    print(f"running {' '.join(common[2:])} straight through, then interrupted at 50")
    subprocess.run(common + ["--out", straight], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(common + ["--out", interrupted, "--stop-after", "50"],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(common + ["--out", interrupted, "--resume"],
                   check=True, stdout=subprocess.DEVNULL)

    curves = []
    for out in (straight, interrupted):
        with open(f"{out}/losses.txt") as f:
            curves.append([float(line.split()[2]) for line in f])
    assert len(curves[0]) == len(curves[1]) == 100, [len(c) for c in curves]
    gap = [abs(a - b) for a, b in zip(*curves)]
    print(f"resume: loss at step 51 {curves[0][50]:.6f} vs {curves[1][50]:.6f}, "
          f"worst gap over 100 steps {max(gap):.2e}, "
          f"final {curves[0][-1]:.6f} vs {curves[1][-1]:.6f}")
    assert max(gap) < 1e-6

def check(args):
    args.device = "cpu"  # fp32 and deterministic; the tolerances assume it
    os.makedirs(OUT, exist_ok=True)
    check_schedule()

    torch.manual_seed(SEED)
    model = Sprout()
    val_data = split(DATA, "val")
    check_eval(model, val_data, args)
    check_accumulation(model, val_data)
    check_resume()

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--stop-after", type=int, default=0,
                   help="halt early; the LR schedule still spans --steps")
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--ctx", type=int, default=CTX)
    p.add_argument("--micro-batch", type=int, default=MICRO_BATCH)
    p.add_argument("--accum", type=int, default=ACCUM)
    p.add_argument("--layers", type=int, default=N_LAYER)
    p.add_argument("--dim", type=int, default=DIM)
    p.add_argument("--heads", type=int, default=N_HEAD)
    p.add_argument("--lr", type=float, default=PEAK_LR)
    p.add_argument("--data", default=DATA, help="directory of tokenizer.model, train.bin, val.bin")
    p.add_argument("--pos", default="rope", choices=["rope", "learned"])
    p.add_argument("--out", default=OUT)
    p.add_argument("--hub", default=HUB)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--toy", action="store_true", help="the TOY shape, unless overridden")
    args = p.parse_args()
    if args.toy:
        for k, v in TOY.items():
            if p.get_default(k) == getattr(args, k):
                setattr(args, k, v)
    return args


if __name__ == "__main__":
    args = parse()
    check(args) if args.check else run(args)
