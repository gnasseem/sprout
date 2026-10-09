// The same byte BPE and Unicode pre-splitting as tokenizer.py.
const pattern = /'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\p{White_Space}\p{L}\p{N}]+[\r\n]*|\p{White_Space}*[\r\n]|\p{White_Space}+(?![^\p{White_Space}])|\p{White_Space}+/gu;
const specials = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"];

export function tokenizer(text) {
  const pairs = text.trim().split(/\r?\n/).map(line => line.split(" ").map(Number));
  const ranks = new Map();
  const vocab = Array.from({ length: 256 }, (_, i) => new Uint8Array([i]));
  for (const [a, b] of pairs) {
    ranks.set(`${a},${b}`, vocab.length);
    vocab.push(new Uint8Array([...vocab[a], ...vocab[b]]));
  }
  const special = Object.fromEntries(specials.map((s, i) => [s, vocab.length + i]));
  const encoder = new TextEncoder();
  const split = /(<\|endoftext\|>|<\|user\|>|<\|assistant\|>|<\|end\|>)/g;
  function encode(text) {
    const result = [];
    for (const part of text.split(split)) {
      if (part in special) {
        result.push(special[part]);
        continue;
      }
      for (const match of part.matchAll(pattern)) {
        let ids = Array.from(encoder.encode(match[0]));
        while (ids.length > 1) {
          let rank = Infinity;
          let a, b;
          for (let i = 0; i < ids.length - 1; i++) {
            const value = ranks.get(`${ids[i]},${ids[i + 1]}`) ?? Infinity;
            if (value < rank) { rank = value; a = ids[i]; b = ids[i + 1]; }
          }
          if (!Number.isFinite(rank)) break;
          const merged = [];
          for (let i = 0; i < ids.length; i++) {
            if (ids[i] === a && ids[i + 1] === b) { merged.push(rank); i++; }
            else merged.push(ids[i]);
          }
          ids = merged;
        }
        result.push(...ids);
      }
    }
    return result;
  }
  return { encode, vocab, special };
}

export function storyPrompt(fields) {
  const names = ["Summary", "Words", "Random sentence", "Features"];
  if (Object.keys(fields).some(key => !names.includes(key)) ||
      Object.values(fields).some(value => typeof value !== "string")) {
    throw new Error("Use the supported story fields with text values.");
  }
  if (Object.values(fields).reduce((n, value) => n + value.length, 0) > 4096) {
    throw new Error("Your instructions are too long. Try a shorter story idea.");
  }
  if (Object.values(fields).some(value => specials.some(s => value.includes(s)))) {
    throw new Error("Remove reserved tokens such as <|user|> from your instructions.");
  }
  const clean = names.map(key => [key, (fields[key] ?? "").replace(/[\p{White_Space}\u001c-\u001f]+/gu, " ").replace(/^ | $/g, "")]);
  const lines = clean.filter(([, value]) => value).map(([key, value]) => `${key}: ${value}`);
  if (!lines.length) throw new Error("Enter a story idea or a word to include.");
  return `<|user|>\nWrite a short story.\n${lines.join("\n")}<|end|>\n<|assistant|>\n`;
}
