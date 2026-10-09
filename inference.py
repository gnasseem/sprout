"""Load a standalone Sprout bundle and stream instruction-tuned stories."""

import hashlib
import json
import math
from pathlib import Path

import torch

import tokenizer
from model import Sprout

TEMPLATE = "<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
FIELDS = ("Summary", "Words", "Random sentence", "Features")


def instruction(fields):
    return "Write a short story.\n" + "\n".join(
        f"{key}: {fields[key]}" for key in FIELDS if fields.get(key))


def prompt(tok, fields):
    return tok.encode(TEMPLATE.format(instruction=instruction(fields)))


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(source, device="cpu", revision=None):
    """source is an exported directory or Hub model ID; revision may pin a commit."""
    from safetensors.torch import load_model

    path = Path(source)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(str(source), revision=revision,
                    allow_patterns=["config.json", "model.safetensors", "tokenizer.model"]))
    cfg = json.loads((path / "config.json").read_text())
    if cfg["format_version"] != 1 or cfg["template"] != TEMPLATE or cfg["fields"] != list(FIELDS):
        raise ValueError("Unsupported model format or instruction template.")
    for name, expected in cfg["sha256"].items():
        if name not in ("model.safetensors", "tokenizer.model") or digest(path / name) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    if set(cfg["sha256"]) != {"model.safetensors", "tokenizer.model"}:
        raise ValueError("Bundle must checksum both weights and tokenizer.")
    tok = tokenizer.load(path / "tokenizer.model")
    if len(tok.vocab) != cfg["model"]["vocab"] or tok.special != cfg["special_tokens"]:
        raise ValueError("Tokenizer does not match the model configuration.")
    model = Sprout(**cfg["model"])
    load_model(model, path / "model.safetensors", strict=True)
    return model.to(device).eval(), tok, cfg


def stream(model, tok, fields, temperature=0.8, top_k=50, max_tokens=256):
    """Yield plain text; UTF-8 bytes are buffered until each character is complete."""
    import codecs

    if set(fields) - set(FIELDS) or any(not isinstance(v, str) for v in fields.values()):
        raise ValueError("Use the supported story fields with text values.")
    if sum(len(v) for v in fields.values()) > 4096:
        raise ValueError("Story instructions are too long. Shorten them and try again.")
    if any(s in v for v in fields.values() for s in tokenizer.SPECIALS):
        raise ValueError("Remove reserved tokens such as <|user|> from the story fields.")
    # Each field occupies one metadata line in the training template.
    fields = {key: " ".join(value.split()) for key, value in fields.items()}
    if not any(fields.values()):
        raise ValueError("Enter a story idea or an optional story constraint.")
    if not math.isfinite(float(temperature)) or not 0 <= temperature <= 2:
        raise ValueError("Temperature must be between 0 and 2; 0 selects greedy decoding.")
    if not isinstance(top_k, int) or not 1 <= top_k <= len(tok.vocab):
        raise ValueError("Top-k must be a positive integer within the vocabulary.")
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= model.ctx:
        raise ValueError(f"Maximum tokens must be between 1 and {model.ctx}.")
    prefix = prompt(tok, fields)
    # Leave room for a story, and keep the entire instruction in view.
    if len(prefix) > model.ctx - 32:
        raise ValueError("Story instructions use too much context. Shorten them and try again.")
    limit = min(max_tokens, model.ctx - len(prefix) + 1)
    x = torch.tensor([prefix], device=next(model.parameters()).device)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = ""
    yield text
    tokens = model.generate_tokens(x, limit, temperature=temperature, top_k=top_k,
                                   stop=tok.special["<|end|>"])
    for token in tokens:
        idx = token.item()
        if idx in tok.special.values():
            break
        part = decoder.decode(tok.vocab[idx])
        if part:
            text += part
            yield text
    tail = decoder.decode(b"", final=True)
    if tail:
        yield text + tail
