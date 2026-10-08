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

`sft.py` prepares [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/tree/ee050ed1f8720795be342921335e821856a2b42e), trains the pretrained model with assistant-only loss, and evaluates word inclusion, exact sentence inclusion, stopping, and base-model validation regression. The full instruction-tuning run requires the A100 and has not run yet.

The exact conversation is `"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{story}<|end|>"`. The instruction starts with `"Write a short story.\n"`, followed by the available `Summary:`, `Words:`, `Random sentence:`, and `Features:` lines in that order, with no trailing newline. `prompt()` produces the identical prefix for training and generation. Labels ignore the prompt and right padding; the last prompt position predicts the first assistant token, and the assistant's final `<|end|>` is supervised. Causal attention prevents right padding from influencing real tokens.

Preparation pins the source revision, parses metadata on either side of `Story:`, and removes malformed records, reserved-token content, and whitespace-equivalent duplicate stories. The official instruction validation stories are excluded from instruction training, as are stories in the existing base validation split. Complete conversations longer than 512 prediction positions are excluded rather than teaching an artificial early ending. This favors shorter stories. The local validation preparation retained 23,833 of 25,027 records (4,592,270 assistant tokens); it excluded 34 malformed records, 1,019 over-context records, and 141 duplicates or base-validation overlaps.

Run on the CUDA machine with the existing data and main checkpoint available; preparation needs space for the 2.66 GB source and its tokenized copy:

```bash
python sft.py --prepare
python sft.py --check
python sft.py --checkpoint checkpoints/main/ckpt.pt --evaluate --micro-batch 8 > checkpoints/main/instruct-eval.txt
python sft.py --checkpoint checkpoints/main/ckpt.pt --hub GeorgeCodesSometimes/sprout-checkpoints
python sft.py --checkpoint checkpoints/sft/best.pt --evaluate --micro-batch 8 > checkpoints/sft/instruct-eval.txt
```

The default is one shuffled epoch, seed 1337, learning rate 3e-5, up to 100 warmup steps, cosine decay to 10% of peak, AdamW with weight decay 0.1 and betas (0.9, 0.95), gradient clipping at 1.0, and 16 sequences × 8 accumulation batches. Accumulation is weighted by assistant tokens, including the final partial batch. Full instruction validation runs before training, every 500 steps, and at epoch boundaries. W&B records losses, gradients, learning rate, throughput, configuration, and Git commit. `best.pt` selects the lowest instruction validation loss; `ckpt.pt` holds the latest optimizer and RNG state. Both upload to the private checkpoint repository's `sft/` directory when `--hub` is supplied. Use the same training command with `--resume` to continue; source checkpoint and prepared-data hashes must match. `--no-wandb` and an omitted `--hub` support local checks.

Evaluation uses the first 20 valid, in-context official validation records containing both words and a sentence constraint, with fixed seeds 1337–1356, temperature 0.8, and top-k 50. Generation stays within the trained context so the instruction remains visible. Outputs retain all token IDs, full continuations, settings, case-insensitive whole-word inclusion, exact sentence inclusion, and whether `<|end|>` was emitted. Run base and tuned evaluation on the same device and precision; compare full base validation loss as well as instruction metrics. Summary fidelity still requires reading the continuations.

The untuned main checkpoint's October 8 MPS float32 baseline included all requested words in 0/20 prompts, the exact requested sentence in 0/20, and emitted the chat end token in 0/20. It often emitted the pretraining end-of-text token and continued with another story. Full base validation remained 1.091364 nats/token and 0.391208 bits/byte. The complete outputs are in `checkpoints/main/instruct-eval.txt`; tuned results remain pending the GPU run.

The inline checks verify the shifted masking boundary, final end label, zero logit gradients at ignored positions, unequal-length accumulation, and Unicode template round-trip. A local three-step training run with a partial final batch reproduced exactly after save/resume. A CPU update of the main pretrained checkpoint reduced a held-out smoke example's loss from 0.977750 to 0.703327; those temporary weights were discarded.
