import hashlib
import json
import os
from huggingface_hub import hf_hub_download

SEED = b"sprout-v1-split"
VAL_FRAC = 0.01
OUT = "data"
SEP = "<|endoftext|>"

def is_val(story):
    """Returns True if the story should be held out for validation, False otherwise."""
    h = hashlib.blake2b(story.encode("utf-8"), digest_size=8, salt=SEED).digest()
    return int.from_bytes(h, "big") % 10000 < VAL_FRAC * 10000

def stories(path):
    """Yield one story at a time. A lone <|endoftext|> line separates them."""
    lines = []
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            if line.rstrip("\r\n") != SEP:
                lines.append(line)
                continue
            story = "".join(lines).strip()
            lines = []
            if story:
                yield story
    story = "".join(lines).strip()
    if story:
        yield story

def main():
    os.makedirs(OUT, exist_ok=True)
    raw = hf_hub_download(
        "roneneldan/TinyStories",
        "TinyStoriesV2-GPT4-train.txt",
        repo_type="dataset",
        local_dir=OUT,
    )

    train = open(f"{OUT}/train.jsonl", "w", encoding="utf-8")
    val = open(f"{OUT}/val.jsonl", "w", encoding="utf-8")
    n = [0, 0]
    nbytes = 0

    for story in stories(raw):
        held_out = is_val(story)
        (val if held_out else train).write(
            json.dumps({"text": story}, ensure_ascii=False) + "\n")
        n[held_out] += 1
        nbytes += len(story.encode("utf-8"))

    train.close()
    val.close()

    total = n[0] + n[1]
    print(f"{total:,} stories, {nbytes / 1e9:.3f} GB")
    print(f"train {n[0]:,}  val {n[1]:,} ({n[1] / total:.2%})")

if __name__ == "__main__":
    main()