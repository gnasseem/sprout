# Sprout

Sprout is a decoder-only language model trained from scratch on TinyStoriesV2-GPT4. The main model has 29,368,832 parameters: 8 layers, width 512, 8 attention heads, a 512-token context, RoPE, RMSNorm, SwiGLU, and tied embeddings. Its byte-level BPE tokenizer has 4,096 tokens, including four reserved special tokens.

The deterministic story-level split contains 537,686,596 training tokens and 5,402,089 validation tokens. Training uses AdamW, bf16 autocast on CUDA, a 300-step warmup followed by cosine decay, and 131,072 tokens per optimizer step.

## Runs

| Run | Training tokens | Full validation cross-entropy | Bits per byte | Training record |
| --- | ---: | ---: | ---: | --- |
| Main pretraining, 12,000 steps | 1,572,864,000 | 1.091364 | 0.391208 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/y5ye7t4k) |
| One-pass 4,096-token RoPE baseline, 4,102 steps | 537,657,344 | 1.180857 | 0.423287 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/i0wis53q) |
| One-pass 8,192-token RoPE, 3,974 steps | 520,880,128 | 1.202189 | 0.417522 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/yf7r4ema) |
| One-pass 4,096-token learned positions, 4,102 steps | 537,657,344 | 1.196941 | 0.429052 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/ul3wfup5) |

Validation scores were recomputed with `eval.py` on October 8, 2026 using PyTorch 2.14.0, MPS, float32, and batches of 8. The original CUDA summaries used bf16 and excluded the final partial validation window; those historical numbers differ slightly. All rows above use the same complete scoring policy.

The baseline finished on October 7, 2026; both other arms finished on October 8. All final checkpoints are verified locally and in the private [checkpoint repository](https://huggingface.co/GeorgeCodesSometimes/sprout-checkpoints): `ckpt.pt` for the main model, and `base/ckpt.pt`, `vocab8192/ckpt.pt`, and `learned-pos/ckpt.pt` for the controlled runs. Cross-entropies should not be compared across tokenizers; the vocabulary comparison uses bits per byte over the same held-out stories.

## Reproduce

Install `torch numpy regex huggingface_hub wandb`, then run `python data.py`, `python tokenizer.py`, and `python tokenizer.py 8192` to create the seeded split and both tokenizations. The data and checkpoint Hub repositories are currently private; authorized collaborators can instead download the prepared data with:

```bash
hf download GeorgeCodesSometimes/sprout-data --repo-type dataset --local-dir .
mv vocab8192 data/vocab8192
```

On a CUDA machine, run `python model.py` to check the model and its one-batch overfit test. To reproduce the two ablation arms, use the same seed, optimizer, and data split as the 4,102-step baseline:

```bash
python train.py --hub GeorgeCodesSometimes/sprout-checkpoints --out checkpoints/vocab8192 --data data/vocab8192 --steps 3974
python train.py --hub GeorgeCodesSometimes/sprout-checkpoints --out checkpoints/learned-pos --pos learned --steps 4102
```

Each controlled arm has a training-token budget approximately equal to one pass through the same raw stories; batches are random windows sampled with replacement. The 8,192-token arm needs fewer steps because its tokenizer compresses the text differently. All three use seed 1337, 131,072 tokens per optimizer step, a 300-step warmup, peak learning rate 1e-3, and cosine decay over the arm's step count. The main model trains for about 2.93 corpus passes and is outside this controlled comparison.

The larger vocabulary lowers full-validation bits per byte by 1.36%, and its sampled curve stays below the baseline throughout training. It also increases parameters from 29,368,832 to 31,465,984 (7.14%) and uses fewer optimizer steps: this comparison matches the text budget, not parameter count or compute, and uses one seed.

Learned absolute positions raise full-validation bits per byte by 1.36%; their sampled curve stays above RoPE throughout training. The tokenizer and optimizer schedule match, with 262,144 additional position-embedding parameters. This result favors RoPE at the one-pass budget; a longer run or different seed could change the comparison.

## Evaluate

`eval.py` loads the checkpoint's architecture, verifies its data/tokenizer fingerprint, and scores consecutive windows of its context length (512 tokens here), including the final partial window. Every token after the first is a target; end-of-story tokens contribute loss and zero text bytes. Both tokenizers cover the same 21,741,948 scored text bytes. Cross-entropy is in nats per token; bits per byte is total negative log-likelihood divided by `ln(2)` and the scored text bytes.

```bash
python eval.py checkpoints/main/ckpt.pt > checkpoints/main/eval.txt
python eval.py checkpoints/base/ckpt.pt > checkpoints/base/eval.txt
python eval.py checkpoints/vocab8192/ckpt.pt --data data/vocab8192 > checkpoints/vocab8192/eval.txt
python eval.py checkpoints/learned-pos/ckpt.pt > checkpoints/learned-pos/eval.txt
```

Each output includes checkpoint SHA-256, training config, data fingerprint, device/precision, validation scores, and all 20 unedited continuations. Sampling uses temperature 0.8, top-k 50, at most 256 new tokens, and seeds 1337–1356, with an end-of-story token before each opening. Every sample records its settings, generated token IDs, and whether it ended or hit the limit. The first sample is generated twice and must match exactly. Seeds reproduce samples on the same software/device; floating-point differences can change samples across devices.

The main model emits the end-of-story token on 17 of the 20 openings before the token limit. Fluent sentences still lose story facts:

- Opening 2 repeats: “You can wear it on your finger or on your finger.”
- Opening 16 starts with Sam's boat sinking, then treats the heavy stone as a way to make the boat faster.
- Opening 20 changes Rose's name to Lily while describing the marble. The one-pass baseline also drifts from Rose to Rosie and Rosely on this opening.
- The 8,192-token model's opening 2 calls a squirrel a bird and has it fly onto Lily's shoulder. Lower bits per byte does not establish better story quality on this ungraded suite.

These are examples from the fixed suite, not a scored quality benchmark. The complete outputs retain successful stories and failures alike.

## Instruction tuning

`sft.py` prepares [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/tree/ee050ed1f8720795be342921335e821856a2b42e), trains the pretrained model with assistant-only loss, and evaluates word inclusion, exact sentence inclusion, stopping, and base-model validation regression. The first full run completed on October 9, 2026: one shuffled epoch, 18,457 optimizer steps, 2,362,399 conversations, and 457,894,671 supervised assistant tokens. The finished [W&B run](https://wandb.ai/george-nasseem/sprout/runs/rb9ndxvw) records commit `48aa20c` and 5,335 seconds of runtime (about 89 minutes).

The exact conversation is `"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{story}<|end|>"`. The instruction starts with `"Write a short story.\n"`, followed by the available `Summary:`, `Words:`, `Random sentence:`, and `Features:` lines in that order, with no trailing newline. `prompt()` produces the identical prefix for training and generation. Labels ignore the prompt and right padding; the last prompt position predicts the first assistant token, and the assistant's final `<|end|>` is supervised. Causal attention prevents right padding from influencing real tokens.

Preparation pins the source revision, parses metadata on either side of `Story:`, and removes malformed records, reserved-token content, and whitespace-equivalent duplicate stories. The official instruction validation stories are excluded from instruction training, as are stories in the existing base validation split. Complete conversations longer than 512 prediction positions are excluded rather than teaching an artificial early ending. This favors shorter stories. Validation preparation retained 23,833 of 25,027 records (4,592,270 assistant tokens); it excluded 34 malformed records, 1,019 over-context records, and 141 duplicates or base-validation overlaps. Training preparation excluded 3,458 malformed records, 97,915 over-context records, and 12,761 duplicates or validation overlaps from 2,476,533 source records.

Run on the CUDA machine with the existing data and main checkpoint available; preparation needs space for the 2.66 GB source and its tokenized copy. Also copy `data/val.jsonl` from the original split to the machine: it is required to exclude base-validation stories and is absent from the prepared-data Hub repository.

```bash
python sft.py --prepare
python sft.py --check
python sft.py --checkpoint checkpoints/main/ckpt.pt --evaluate --micro-batch 8 > checkpoints/main/instruct-eval.txt
python sft.py --checkpoint checkpoints/main/ckpt.pt --hub GeorgeCodesSometimes/sprout-checkpoints
python sft.py --checkpoint checkpoints/sft/best.pt --evaluate --micro-batch 8 > checkpoints/sft/instruct-eval.txt
```

The default is one shuffled epoch, seed 1337, learning rate 3e-5, up to 100 warmup steps, cosine decay to 10% of peak, AdamW with weight decay 0.1 and betas (0.9, 0.95), gradient clipping at 1.0, and 16 sequences × 8 accumulation batches. Accumulation is weighted by assistant tokens, including the final partial batch. Full instruction validation runs before training, every 500 steps, and at epoch boundaries. W&B records losses, gradients, learning rate, throughput, configuration, and Git commit. `best.pt` selects the lowest instruction validation loss; `ckpt.pt` holds the latest optimizer and RNG state. Both upload to the private checkpoint repository's `sft/` directory when `--hub` is supplied. Use the same training command with `--resume` to continue; source checkpoint and prepared-data hashes must match. `--no-wandb` and an omitted `--hub` support local checks.

Evaluation uses the first 20 valid, in-context official validation records containing both words and a sentence constraint, with fixed seeds 1337–1356, temperature 0.8, and top-k 50. Generation stays within the trained context so the instruction remains visible. Outputs retain all token IDs, full continuations, settings, case-insensitive whole-word inclusion, exact sentence inclusion, and whether `<|end|>` was emitted. Run base and tuned evaluation on the same device and precision; compare full base validation loss as well as instruction metrics. Summary fidelity still requires reading the continuations.

On October 9, both models were evaluated on CUDA with PyTorch `2.11.0+cu128`, bf16 autocast, and batches of 8, using the identical 20 prompts and sampling settings:

| Metric | Pretrained main model | Instruction-tuned model |
| --- | ---: | ---: |
| All requested words included | 0/20 | 6/20 |
| Exact requested sentence included | 1/20 | 4/20 |
| Assistant `<\|end\|>` emitted | 0/20 | 19/20 |
| Full base-validation loss, nats/token | 1.091467 | 1.220414 |
| Full base-validation bits per byte | 0.391245 | 0.437467 |

Instruction validation loss fell from 1.419628 before tuning to 0.956147 at step 18,457, which is both the final and best checkpoint. Stopping improved strongly, but most word and sentence constraints still fail. Base-validation loss rose by 11.81%; this run demonstrates a tradeoff rather than reliable instruction following. The CUDA baseline differs slightly from the earlier MPS float32 evaluation, which scored 0/20 on all three behavioral metrics; comparisons above use the matching CUDA evaluation.

Complete outputs are retained locally in `checkpoints/main/instruct-eval-cuda.txt` and `checkpoints/sft/instruct-eval.txt`, with preparation and training logs in `checkpoints/sft-prepare.txt`, `checkpoints/sft.log`, and `checkpoints/sft/losses.txt`. Both local checkpoints match the SHA-256 hashes of `sft/best.pt` and `sft/ckpt.pt` in the private [checkpoint repository](https://huggingface.co/GeorgeCodesSometimes/sprout-checkpoints). The best checkpoint also loads with the original tokenizer/data fingerprint and generates on CPU. A standalone export and streaming Gradio demo are implemented below; neither has been published publicly.

The inline checks verify the shifted masking boundary, final end label, zero logit gradients at ignored positions, unequal-length accumulation, and Unicode template round-trip. A local three-step training run with a partial final batch reproduced exactly after save/resume. A CPU update of the main pretrained checkpoint reduced a held-out smoke example's loss from 0.977750 to 0.703327; those temporary weights were discarded.

## Cached generation

`model.generate()` uses a KV cache by default; `use_cache=False` runs the full-prefix reference. Set `temperature=0` for greedy decoding. `model.generate_tokens()` yields one `(batch, 1)` token tensor at a time for streaming. Each iterator owns its cache, and finished batch rows emit only the stop token while the other rows continue.

`model.forward_cached(tokens, cache=None)` returns chunk logits and a tuple of per-layer `(K, V)` tensors, each shaped `(batch, heads, cached_tokens, head_dim)`. RoPE and learned positions start at the cached prefix length, and multi-token chunks use an offset causal mask. Caching requires `model.eval()` and runs without gradients. Training and checkpoint weights use the same format as before.

At the 512-token context limit, generation rebuilds the cache from the last 512 tokens at position zero. Simply dropping the oldest keys would retain hidden states from a different context and change predictions. This rebuild matches uncached sliding-window generation but loses the speedup once the window is full. Instruction generation already limits the continuation to keep the complete prompt visible.

```bash
python model.py --check-cache --device cpu
python model.py --check-cache --device mps
```

The checks passed on CPU and MPS and cover 224 exact greedy token matches for both position encodings, context rollover, chunk logits, streaming, and batched stopping. The main pretrained checkpoint also matched 256 greedy tokens exactly on both devices in float32, plus 24 tokens from a 500-token prefix across rollover on CPU. On this laptop with four CPU threads, the 256-token run took 6.234 seconds uncached and 1.779 seconds cached (3.50× faster).

## Portable inference and demo

Export the best instruction checkpoint into a new, empty directory and run the CPU demo:

```bash
python -m pip install -r requirements.txt
python export.py --checkpoint checkpoints/sft/best.pt --out checkpoints/release
python app.py --model checkpoints/release
```

Open `http://127.0.0.1:7860`. The model loads once at startup; requests stream plain text through its KV cache. Story idea, words, sentence, and features use the exact training template shared with `sft.py`. Temperature, top-k, a generation limit, and cancellation are exposed. Requests run serially with a bounded queue. Empty fields, reserved-token injection, and oversized instructions produce actionable errors; instructions remain visible throughout generation. UTF-8 byte fragments are buffered until a character is complete.

`export.py` verifies the checkpoint/data fingerprint before writing float32 safetensors, architecture and checksum configuration, the original tokenizer merges, the inference source files, dependencies, license, and model card. Optimizer state and training data are excluded. The tied embedding/output weight is stored once and restored as a shared tensor. The export checks 64 exact greedy continuation tokens against the checkpoint. The default weights occupy 117,482,344 bytes.

The whole bundle can be copied to a machine without the training files. From inside that copied directory:

```python
from inference import load, stream

model, tok, config = load(".")
for text in stream(model, tok, {"Summary": "A fox finds a seed."}):
    print(text, end="\r", flush=True)
```

`load` also accepts a Hugging Face model ID with an optional `revision` commit. It downloads only weights, configuration, and tokenizer files, verifies both SHA-256 checksums, and uses the included local Python implementation. This is a custom PyTorch loading contract, not a Transformers `AutoModel` integration. See the [Hub upload documentation](https://huggingface.co/docs/hub/models-uploading) and [safetensors tied-weight documentation](https://huggingface.co/docs/safetensors/torch_shared_tensors).

For a model repository, upload the exported directory with its generated `README.md`. For a Gradio Space, upload the same files but use `SPACE.md` as the Space's `README.md`; its frontmatter pins Gradio 6.30.0 and Python 3.11. The app honors Gradio's server-address environment setting; `python app.py --host 0.0.0.0` supports other containers. The [Spaces runtime](https://huggingface.co/docs/hub/spaces-sdks-gradio) installs the listed dependencies and runs `app.py`. No upload is performed by the export or demo commands.

On October 9, 2026, an isolated, freshly installed environment with PyTorch 2.14.1 and Gradio 6.30.0 loaded the copied bundle offline, generated and streamed on CPU, built the queued demo, and rejected a modified tokenizer. Browser generation and cancellation passed at 1440-pixel desktop and 390-pixel mobile widths without horizontal overflow. HTTP checks covered omitted optional fields, Unicode and nonsense prompts, and actionable errors for empty, injected, and oversized inputs. The existing template/masking checks also passed after sharing the prompt implementation. Hub download and hosted Space startup still require publication. The final project write-up still needs the architecture and tokenizer breakdowns, complete cost accounting, loss-curve plots, and limitations consolidated with exact reproduction commands.
