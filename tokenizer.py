import json
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import regex

VOCAB_SIZE = 4096
SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"]
N_MERGES = VOCAB_SIZE - 256 - len(SPECIALS)
SAMPLE_MB = 30
STRIDE = 64
OUT = "data"

# GPT-4's pre-split pattern: text is cut into chunks first, so no merge can
# span a word boundary.
PAT = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
SPECIAL_PAT = regex.compile("(" + "|".join(regex.escape(s) for s in SPECIALS) + ")")

def merge(ids, pair, new):
    """Replace every occurrence of pair in ids with the single token new."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text):
    """Learn N_MERGES pairs, in the order learned."""
    # Chunks are deduplicated and weighted by count: identical chunks always
    # merge identically, and 30MB holds only ~50k distinct ones.
    freq = Counter(PAT.findall(text))
    words = [list(w.encode("utf-8")) for w in freq]
    reps = list(freq.values())

    count = defaultdict(int)  # pair -> how often it occurs
    where = defaultdict(set)  # pair -> which words contain it
    for i, w in enumerate(words):
        for p, k in Counter(zip(w, w[1:])).items():
            count[p] += k * reps[i]
            where[p].add(i)

    merges = []
    for new in range(256, 256 + N_MERGES):
        best = max(count, key=lambda p: (count[p], -p[0], -p[1]))  # ties: smallest pair
        merges.append(best)
        # Only the words containing best change: drop their pairs, merge, put
        # the new pairs back.
        for i in where.pop(best):
            w, r = words[i], reps[i]
            for p, k in Counter(zip(w, w[1:])).items():
                count[p] -= k * r
                if count[p] == 0:
                    del count[p]
            words[i] = w = merge(w, best, new)
            for p, k in Counter(zip(w, w[1:])).items():
                count[p] += k * r
                where[p].add(i)
    return merges

class Tokenizer:
    def __init__(self, merges):
        self.ranks = {p: 256 + i for i, p in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(merges)
        self.special = {s: base + i for i, s in enumerate(SPECIALS)}
        for s, i in self.special.items():
            self.vocab[i] = s.encode("utf-8")
        self.cache = {}  # chunk -> ids, looked up ~400M times over the corpus

    def encode_chunk(self, chunk):
        """Apply the merges to one chunk, earliest-learned pair first."""
        ids = self.cache.get(chunk)
        if ids is not None:
            return ids
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 1 << 30))
            if pair not in self.ranks:
                break
            ids = merge(ids, pair, self.ranks[pair])
        self.cache[chunk] = ids
        return ids

    def encode(self, text):
        # Specials are split out first, so BPE never sees them and cannot
        # shred one into pieces.
        ids = []
        for part in SPECIAL_PAT.split(text):
            if part in self.special:
                ids.append(self.special[part])
            else:
                ids += [i for c in PAT.findall(part) for i in self.encode_chunk(c)]
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

def save(merges, path):
    with open(path, "w") as f:
        f.writelines(f"{a} {b}\n" for a, b in merges)

def load(path=f"{OUT}/tokenizer.model"):
    with open(path) as f:
        return Tokenizer([tuple(int(x) for x in line.split()) for line in f])

def sample(path, mb, stride):
    """Every stride-th story, up to mb megabytes, so the sample spans the corpus."""
    parts, n = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            parts.append(json.loads(line)["text"])
            n += len(parts[-1])
            if n > mb * 1e6:
                break
    return "\n".join(parts)

def batches(path, n=2000):
    """Stories from a .jsonl file, n at a time."""
    batch = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            batch.append(json.loads(line)["text"])
            if len(batch) == n:
                yield batch
                batch = []
    if batch:
        yield batch

enc = None  # one Tokenizer per worker process

def init_worker(merges):
    global enc
    enc = Tokenizer(merges)

def encode_batch(batch):
    eot = enc.special["<|endoftext|>"]
    ids, nbytes = [], 0
    for s in batch:
        story = enc.encode(s)
        assert enc.decode(story) == s, s[:80]  # lossless only if every char lands in a chunk
        ids += story + [eot]
        nbytes += len(s.encode("utf-8"))
    return np.array(ids, dtype=np.uint16), nbytes, len(ids) - len(batch)

def tokenize(name, merges):
    """Write data/<name>.bin, return its byte and token counts."""
    ntok = nbytes = 0
    with Pool(initializer=init_worker, initargs=(merges,)) as pool:
        with open(f"{OUT}/{name}.bin", "wb") as out:
            for arr, nb, nt in pool.imap(encode_batch, batches(f"{OUT}/{name}.jsonl")):
                out.write(arr.tobytes())
                nbytes += nb
                ntok += nt
    return nbytes, ntok

def main():
    text = sample(f"{OUT}/train.jsonl", SAMPLE_MB, STRIDE)
    print(f"training on {len(text) / 1e6:.1f}MB, {len(set(PAT.findall(text))):,} chunks")
    merges = train(text)
    save(merges, f"{OUT}/tokenizer.model")

    tok = load()
    assert len(tok.vocab) == VOCAB_SIZE, len(tok.vocab)
    word = tok.encode("story")
    for s in SPECIALS:
        assert tok.encode(f"story{s}story") == word + [tok.special[s]] + word, s

    step = N_MERGES // 18
    print("vocab:", ", ".join(repr(tok.vocab[i].decode("utf-8", "replace"))
                              for i in range(256, 256 + N_MERGES, step)))

    for name in ("val", "train"):
        nbytes, ntok = tokenize(name, merges)
        print(f"{name}: {ntok:,} tokens, {nbytes / ntok:.3f} bytes/token")

    ids = np.memmap(f"{OUT}/train.bin", dtype=np.uint16, mode="r")[:220]
    print("\n" + tok.decode(ids.tolist()))


if __name__ == "__main__":
    main()
