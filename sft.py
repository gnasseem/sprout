"""Instruction tuning with assistant-only loss and the same chat format at inference.

    python sft.py --prepare
    python sft.py --check
    python sft.py --checkpoint checkpoints/main/ckpt.pt
    python sft.py --checkpoint checkpoints/sft/ckpt.pt --evaluate
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import subprocess
import time

import numpy as np
import torch

import tokenizer
from data import stories
from eval import load
from inference import FIELDS, TEMPLATE, instruction, prompt
from model import Sprout
from train import (BETAS, DEVICE, GRAD_CLIP, SEED, WEIGHT_DECAY, autocast, fingerprint, lr_at,
                   optimizer, split, token_bytes, val_metrics)

REPO = "roneneldan/TinyStoriesInstruct"
REVISION = "ee050ed1f8720795be342921335e821856a2b42e"
FIELD = re.compile(r"^(Summary|Words|Random sentence|Features):\s*(.*)$")


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def parse(text):
    """Headers may occur before Story: or after the story's final paragraph."""
    fields, body, in_story = {}, [], False
    for line in text.splitlines():
        match = FIELD.fullmatch(line)
        if match:
            key, value = match.groups()
            if key in fields:
                raise ValueError(f"duplicate field: {key}")
            fields[key] = value.strip()
        elif line.startswith("Story:"):
            if in_story:
                raise ValueError("duplicate Story header")
            in_story = True
            body.append(line[len("Story:"):].lstrip())
        elif in_story:
            body.append(line)
        elif line.strip():
            raise ValueError("text before Story header")
    story = "\n".join(body).strip()
    if not story or not any(fields.values()):
        raise ValueError("missing story or instruction")
    if any(s in story or any(s in v for v in fields.values()) for s in tokenizer.SPECIALS):
        raise ValueError("reserved token in content")
    return fields, story


def encode(tok, fields, story):
    prefix = prompt(tok, fields)
    ids = prefix + tok.encode(story) + [tok.special["<|end|>"]]
    assert tok.decode(ids) == TEMPLATE.format(instruction=instruction(fields)) + story + "<|end|>"
    return ids, len(prefix)


def story_hash(text):
    # Collapse whitespace only for deduplication, never for training text.
    return hashlib.sha256(" ".join(text.split()).encode()).digest()


def format_hash(data):
    # The binary index carries its encoding contract, so cached tokens cannot
    # silently be used with another tokenizer or a changed conversation format.
    contract = [digest(f"{data}/tokenizer.model"), TEMPLATE, FIELDS, REVISION]
    return hashlib.sha256(json.dumps(contract).encode()).digest()


def prepare(args):
    from huggingface_hub import hf_hub_download
    os.makedirs(args.instruct, exist_ok=True)
    tok = tokenizer.load(f"{args.data}/tokenizer.model")
    excluded = set()
    with open(f"{args.data}/val.jsonl") as f:
        for line in f:
            excluded.add(story_hash(json.loads(line)["text"]))
    # Process validation first so its stories cannot also enter training.
    seen = set()
    for name in ("val", "train"):
        if args.prepare_split and name != args.prepare_split:
            continue
        source = "valid" if name == "val" else "train"
        raw = hf_hub_download(REPO, f"TinyStories-Instruct-{source}.txt",
                               repo_type="dataset", revision=REVISION,
                               local_dir=args.instruct)
        counts, offset = Counter(), 0
        path = f"{args.instruct}/{name}"
        with open(path + ".bin.tmp", "wb") as tokens, open(path + ".idx.tmp", "wb") as index:
            index.write(format_hash(args.data))
            for text in stories(raw):
                counts["source"] += 1
                try:
                    fields, story = parse(text)
                except ValueError:
                    counts["malformed"] += 1
                    continue
                key = story_hash(story)
                if key in excluded or key in seen:
                    counts["duplicate_or_base_val"] += 1
                    continue
                seen.add(key)
                ids, prefix = encode(tok, fields, story)
                if len(ids) - 1 > args.ctx:
                    counts["over_context"] += 1
                    continue
                np.asarray(ids, dtype=np.uint16).tofile(tokens)
                np.asarray([offset, len(ids), prefix], dtype=np.uint64).tofile(index)
                offset += len(ids)
                counts["kept"] += 1
                counts["assistant_tokens"] += len(ids) - prefix
                if counts["source"] % 10000 == 0:
                    print(name, dict(counts), flush=True)
        assert counts["kept"], counts
        os.replace(path + ".bin.tmp", path + ".bin")
        os.replace(path + ".idx.tmp", path + ".idx")
        print(name, dict(counts), flush=True)


def dataset(root, name, data="data"):
    with open(f"{root}/{name}.idx", "rb") as f:
        assert f.read(32) == format_hash(data), "prepared data encoding differs; run --prepare"
    ids = np.memmap(f"{root}/{name}.bin", mode="r", dtype=np.uint16)
    index = np.memmap(f"{root}/{name}.idx", mode="r", dtype=np.uint64, offset=32).reshape(-1, 3)
    assert len(index) and int(index[-1, 0] + index[-1, 1]) == len(ids)
    return ids, index


def collate(data, indices, pad, device):
    ids, rows = data
    selected = rows[indices]
    width = int(selected[:, 1].max()) - 1
    x = torch.full((len(indices), width), pad, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (offset, length, prefix) in enumerate(selected.tolist()):
        assert 0 < prefix < length
        seq = torch.from_numpy(ids[offset:offset + length].astype(np.int64))
        x[i, :length - 1] = seq[:-1]
        # Logit at prefix-1 predicts the FIRST assistant token. The last
        # non-padding label is the assistant end token.
        y[i, prefix - 1:length - 1] = seq[prefix:]
    return x.to(device), y.to(device)


@torch.no_grad()
def validation(model, data, args, pad):
    was_training = model.training
    model.eval()
    nats = targets = 0
    for start in range(0, len(data[1]), args.micro_batch):
        indices = np.arange(start, min(start + args.micro_batch, len(data[1])))
        x, y = collate(data, indices, pad, args.device)
        n = int((y != -100).sum())
        with autocast(args.device):
            loss = model(x, y)[1]
        nats += loss.item() * n
        targets += n
    model.train(was_training)
    return nats / targets


def save(path, model, opt, cfg, fp, step, best, run_id):
    state = dict(model=model.state_dict(), optimizer=opt.state_dict(), config=cfg["model"],
                 fingerprint=fp, step=step, sft=cfg, best_val=best, wandb_id=run_id,
                 torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 mps_rng=torch.mps.get_rng_state() if next(model.parameters()).device.type == "mps" else None)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def upload(path, hub):
    if hub:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=path, path_in_repo="sft/" + os.path.basename(path),
                           repo_id=hub, commit_message="Instruction tuning checkpoint")


def train(args):
    assert args.epochs in (1, 2, 3) and 0 < args.lr <= 5e-5
    assert args.micro_batch > 0 and args.accum > 0 and args.eval_every > 0
    torch.manual_seed(SEED)
    if args.device == "mps":
        torch.mps.manual_seed(SEED)
    model, tok, model_cfg, _ = load(args.checkpoint, args.data, args.device)
    train_data = dataset(args.instruct, "train", args.data)
    val_data = dataset(args.instruct, "val", args.data)
    assert max(train_data[1][:, 1].max(), val_data[1][:, 1].max()) - 1 <= model.ctx
    pad = tok.special["<|endoftext|>"]
    size = args.micro_batch * args.accum
    per_epoch = math.ceil(len(train_data[1]) / size)
    steps = per_epoch * args.epochs
    cfg = dict(model=model_cfg, template=TEMPLATE, fields=FIELDS, revision=REVISION,
               checkpoint_sha256=digest(args.checkpoint), seed=SEED, epochs=args.epochs,
               micro_batch=args.micro_batch, accum=args.accum, lr=args.lr,
               weight_decay=WEIGHT_DECAY, betas=BETAS, grad_clip=GRAD_CLIP,
               eval_every=args.eval_every, device=args.device,
               warmup=min(100, max(1, steps // 20)), steps=steps,
               data={f"{name}.{ext}": digest(f"{args.instruct}/{name}.{ext}")
                     for name in ("train", "val") for ext in ("bin", "idx")})
    fp = fingerprint(args.data)
    opt = optimizer(model, args.device)
    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/ckpt.pt"
    step, best, run_id = 0, float("inf"), None
    if args.resume:
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["sft"] == cfg and state["fingerprint"] == fp
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step, best, run_id = state["step"], state["best_val"], state["wandb_id"]
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            assert args.device == "cuda", "resume on the original device"
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state["mps_rng"] is not None:
            assert args.device == "mps", "resume on the original device"
            torch.mps.set_rng_state(state["mps_rng"])
    wandb = None
    if not args.no_wandb:
        import wandb
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        run = wandb.init(project="sprout", name=os.path.basename(args.out), id=run_id,
                         resume="must" if run_id else None, config=cfg | {"commit": commit})
        run_id = run.id
    print(json.dumps(cfg), flush=True)
    model.train()
    order, epoch = None, -1
    stop = min(steps, args.stop_after or steps)
    with open(f"{args.out}/losses.txt", "a" if args.resume else "w") as log:
        if step == 0:
            initial = validation(model, val_data, args, pad)
            print(f"step 0 sft_val_loss {initial:.6f}", flush=True)
            print(json.dumps(dict(step=0, sft_val_loss=initial)), file=log, flush=True)
            if wandb:
                wandb.log({"sft_val_loss": initial}, step=0)
        while step < stop:
            current_epoch, batch_id = divmod(step, per_epoch)
            if current_epoch != epoch:
                epoch = current_epoch
                order = torch.randperm(len(train_data[1]), generator=torch.Generator().manual_seed(SEED + epoch)).numpy()
            rows = order[batch_id * size:(batch_id + 1) * size]
            # Weight accumulation by supervised tokens, not padded batch size.
            target_count = int((train_data[1][rows, 1] - train_data[1][rows, 2]).sum())
            lr = lr_at(step, steps, cfg["warmup"], args.lr)
            for group in opt.param_groups:
                group["lr"] = lr
            t0, total = time.time(), 0.0
            opt.zero_grad(set_to_none=True)
            for start in range(0, len(rows), args.micro_batch):
                x, y = collate(train_data, rows[start:start + args.micro_batch], pad, args.device)
                weight = int((y != -100).sum()) / target_count
                with autocast(args.device):
                    loss = model(x, y)[1]
                (loss * weight).backward()
                total += loss.item() * weight
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
            opt.step()
            step += 1
            metrics = dict(step=step, loss=total, lr=lr, grad_norm=float(grad),
                           assistant_tokens_per_sec=target_count / (time.time() - t0))
            print(json.dumps(metrics), flush=True)
            print(json.dumps(metrics), file=log, flush=True)
            if wandb:
                wandb.log(metrics, step=step)
            if step % args.eval_every == 0 or step % per_epoch == 0 or step == stop:
                val = validation(model, val_data, args, pad)
                print(json.dumps(dict(step=step, sft_val_loss=val)), file=log, flush=True)
                print(f"step {step} sft_val_loss {val:.6f}", flush=True)
                if wandb:
                    wandb.log({"sft_val_loss": val}, step=step)
                if val < best:
                    best = val
                    save(f"{args.out}/best.pt", model, opt, cfg, fp, step, best, run_id)
                    upload(f"{args.out}/best.pt", args.hub)
                save(path, model, opt, cfg, fp, step, best, run_id)
                upload(path, args.hub)
    if wandb:
        wandb.finish()


@torch.inference_mode()
def evaluate(args):
    model, tok, cfg, step = load(args.checkpoint, args.data, args.device)
    args.ctx = cfg["ctx"]
    loss, bpb = val_metrics(model, split(args.data, "val"), token_bytes(tok), args)
    print(json.dumps(dict(checkpoint=args.checkpoint, sha256=digest(args.checkpoint),
                          step=step, device=args.device, torch=torch.__version__,
                          dtype="bfloat16" if args.device == "cuda" else "float32",
                          template=TEMPLATE, revision=REVISION, micro_batch=args.micro_batch,
                          base_val_loss=loss, base_val_bpb=bpb)), flush=True)
    raw = f"{args.instruct}/TinyStories-Instruct-valid.txt"
    records = []
    for text in stories(raw):
        try:
            fields, story = parse(text)
        except ValueError:
            continue
        if len(encode(tok, fields, story)[0]) - 1 <= model.ctx and fields.get("Words") and fields.get("Random sentence"):
            records.append(fields)
        if len(records) == 20:
            break
    assert len(records) == 20, "need 20 held-out word-and-sentence prompts"
    totals = Counter()
    for i, fields in enumerate(records):
        torch.manual_seed(SEED + i)
        if args.device == "mps":
            torch.mps.manual_seed(SEED + i)
        prefix = prompt(tok, fields)
        # Stay inside training context so the instruction remains visible.
        limit = model.ctx - len(prefix) + 1
        end = tok.special["<|end|>"]
        x = torch.tensor([prefix], device=args.device)
        with autocast(args.device):
            ids = model.generate(x, limit, temperature=0.8, stop=end, top_k=50)[0, len(prefix):].tolist()
        ended = bool(ids and ids[-1] == end)
        result = tok.decode(ids[:-1] if ended else ids)
        words = [w.strip() for w in fields["Words"].split(",")]
        included = [bool(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", result, re.I)) for w in words]
        sentence = fields["Random sentence"] in result
        totals.update(dict(prompts=1, all_words=int(all(included)), sentence=int(sentence), ended=int(ended)))
        print(json.dumps(dict(prompt_id=i + 1, fields=fields, continuation=result,
                              token_ids=ids, seed=SEED + i, temperature=0.8, top_k=50,
                              max_tokens=limit, use_cache=True, words_included=included,
                              sentence_included=sentence, ended=ended), ensure_ascii=False), flush=True)
    print(json.dumps(dict(behavior=dict(totals))), flush=True)


def check(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tok = tokenizer.load(f"{args.data}/tokenizer.model")
    fields = {"Summary": "A café cat smiles.", "Words": "cat, happy", "Random sentence": "The cat is happy."}
    pairs = [(fields, "The cat is happy."), ({"Words": "hi"}, "Hi!")]
    flat, rows = [], []
    for f, story in pairs:
        ids, prefix = encode(tok, f, story)
        assert ids[:prefix] == prompt(tok, f)
        rows.append([len(flat), len(ids), prefix])
        flat.extend(ids)
    data = np.array(flat, dtype=np.uint16), np.array(rows, dtype=np.uint64)
    x, y = collate(data, np.arange(2), tok.special["<|endoftext|>"], "cpu")
    for i, (offset, length, prefix) in enumerate(rows):
        assert (y[i, :prefix - 1] == -100).all()
        assert y[i, prefix - 1].item() == flat[offset + prefix]
        assert y[i, length - 2].item() == tok.special["<|end|>"]
        assert (y[i, length - 1:] == -100).all()
        assert int((y[i] != -100).sum()) == length - prefix
    model = Sprout(n_layer=1, dim=32, n_head=4, ctx=x.shape[1], vocab=len(tok.vocab))
    logits, loss = model(x, y)
    logits.retain_grad()
    manual = torch.nn.functional.cross_entropy(logits[y != -100], y[y != -100])
    assert torch.allclose(loss, manual)
    manual.backward()
    assert (logits.grad[y == -100] == 0).all()
    full_grad = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for i in range(2):
        xx, yy = collate(data, [i], tok.special["<|endoftext|>"], "cpu")
        (model(xx, yy)[1] * (yy != -100).sum() / (y != -100).sum()).backward()
    error = max((p.grad - g).abs().max().item() for p, g in zip(model.parameters(), full_grad))
    assert error < 1e-6, error
    parsed, story = parse("Words: cat, happy\nStory: \nThe cat is happy.\nSummary: A happy cat.\n")
    assert story == "The cat is happy." and parsed["Summary"] == "A happy cat."
    print(f"template round-trip, first assistant label, end label, padding and prompt gradients passed; accumulation max error {error:.2e}")
    print(f"supervised tokens: {(y != -100).sum().item()}; ignored labels: {(y == -100).sum().item()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--instruct", default="data/instruct")
    p.add_argument("--checkpoint", default="checkpoints/main/ckpt.pt")
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default=DEVICE)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--hub", default="")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--prepare-split", choices=("val",))
    p.add_argument("--check", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    args = p.parse_args()
    if args.prepare:
        prepare(args)
    elif args.check:
        check(args)
    elif args.evaluate:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
