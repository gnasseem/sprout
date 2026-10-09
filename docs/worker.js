import { tokenizer, storyPrompt } from "./tokenizer.js";

const RUNTIME = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
// Pin the model commit so a later upload cannot silently change this demo.
const MODEL = "https://huggingface.co/GeorgeCodesSometimes/sprout-instruct/resolve/ecb41dc46edcfb540d13ad2fe3e72bd863aca25a/browser/";
let ort, session, tok, config;
let stopped = false;
let busy = false;
const send = (type, data = {}) => self.postMessage({ type, ...data });

async function response(name) {
  const result = await fetch(MODEL + name);
  if (!result.ok) throw new Error(`Model download failed (${result.status}). Check your connection and try again.`);
  return result;
}

async function load(onlyTokenizer = false) {
  if (session) return;
  if (!tok) {
    send("status", { text: "Downloading model settings…" });
    config = await (await response("config.json")).json();
    const source = await (await response("tokenizer.model")).text();
    const tokenHash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
    const checksum = Array.from(new Uint8Array(tokenHash), b => b.toString(16).padStart(2, "0")).join("");
    if (checksum !== config.sha256["tokenizer.model"]) throw new Error("Tokenizer checksum failed. Reload and try again.");
    const parsed = tokenizer(source);
    if (parsed.vocab.length + 4 !== config.model.vocab ||
        JSON.stringify(parsed.special) !== JSON.stringify(config.special_tokens)) {
      throw new Error("Model and tokenizer do not match. Reload the page and try again.");
    }
    tok = parsed;
  }
  if (onlyTokenizer) return;
  let storage;
  try { storage = await caches.open("sprout-browser-v1"); } catch { /* Cache is optional. */ }
  let file = await storage?.match(MODEL + "model.onnx");
  if (!file) {
    file = await response("model.onnx");
    if (storage) storage.put(MODEL + "model.onnx", file.clone()).catch(() => {});
  }
  const bytes = new Uint8Array(config.browser_bytes);
  const reader = file.body.getReader();
  let offset = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.length > bytes.length) throw new Error("Model download has an unexpected size. Reload and try again.");
    bytes.set(value, offset);
    offset += value.length;
    send("progress", { value: offset / bytes.length });
  }
  if (offset !== bytes.length) throw new Error("Model download was interrupted. Reload and try again.");
  send("status", { text: "Checking weights and starting the model…" });
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  const checksum = Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, "0")).join("");
  if (checksum !== config.browser_sha256) {
    await storage?.delete(MODEL + "model.onnx");
    throw new Error("Model checksum failed. Reload to download a fresh copy.");
  }
  ort = await import(RUNTIME + "ort.wasm.min.mjs");
  // One worker thread also works on hosts without cross-origin isolation.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = RUNTIME;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  send("loaded");
}

function sample(logits, temperature) {
  if (temperature === 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const sorted = Array.from(logits, (value, id) => ({ value, id })).sort((a, b) => b.value - a.value);
  const cutoff = sorted[49].value;
  const candidates = sorted.filter(item => item.value >= cutoff);
  const weights = candidates.map(item => Math.exp((item.value - sorted[0].value) / temperature));
  let remaining = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < weights.length; i++) {
    remaining -= weights[i];
    if (remaining < 0) return candidates[i].id;
  }
  return candidates.at(-1).id;
}

async function generate({ fields, temperature, maxTokens }) {
  // Reject invalid instructions before starting the large download.
  const prefix = storyPrompt(fields);
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2 ||
      !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 512) {
    throw new Error("Choose a valid temperature and story length.");
  }
  await load(true);
  let ids = tok.encode(prefix);
  const { n_layer: layers, n_head: heads, dim, ctx } = config.model;
  if (ids.length > ctx - 32) throw new Error("Your instructions leave too little room for a story. Shorten them and try again.");
  await load();
  if (stopped) { send("done", { reason: "Stopped", tokens: 0 }); return; }
  const limit = Math.min(maxTokens, ctx - ids.length + 1);
  let cache = new ort.Tensor("float32", new Float32Array(0), [layers, 2, 1, heads, 0, dim / heads]);
  const decoder = new TextDecoder();
  let text = "";
  let count = 0;
  let reason = "Length limit reached";
  const started = performance.now();
  send("status", { text: "Writing your story…" });
  try {
    for (let i = 0; i < limit; i++) {
      if (stopped) { reason = "Stopped"; break; }
      const tokens = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
      let output;
      try { output = await session.run({ tokens, cache }); }
      finally { tokens.dispose(); }
      cache.dispose();
      cache = output.present;
      const next = sample(output.logits.data, temperature);
      output.logits.dispose();
      count++;
      if (next >= tok.vocab.length) {
        reason = next === tok.special["<|end|>"] ? "Story finished" : "Model emitted a reserved token";
        break;
      }
      text += decoder.decode(tok.vocab[next], { stream: true });
      send("text", { text, tokens: count });
      ids = [next];
      // Yield between steps so stop messages are processed even during CPU inference.
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    text += decoder.decode();
    send("text", { text, tokens: count });
    send("done", { reason, tokens: count, seconds: (performance.now() - started) / 1000 });
  } finally { cache.dispose(); }
}

self.onmessage = async ({ data }) => {
  if (data.type === "stop") { stopped = true; return; }
  if (data.type !== "generate" || busy) return;
  busy = true;
  stopped = false;
  try { await generate(data); }
  catch (error) {
    console.error(error);
    send("error", { text: error.message || "The model could not start. Reload and try again." });
  } finally { busy = false; }
};
