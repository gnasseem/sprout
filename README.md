# Sprout

Sprout is a decoder-only language model trained from scratch on TinyStoriesV2-GPT4. The main model has 29,368,832 parameters: 8 layers, width 512, 8 attention heads, a 512-token context, RoPE, RMSNorm, SwiGLU, and tied embeddings. Its byte-level BPE tokenizer has 4,096 tokens, including four reserved special tokens.

The deterministic story-level split contains 537,686,596 training tokens and 5,402,089 validation tokens. Training uses AdamW, bf16 autocast on CUDA, a 300-step warmup followed by cosine decay, and 131,072 tokens per optimizer step.

## Runs

| Run | Training tokens | Validation result | Record |
| --- | ---: | --- | --- |
| Main pretraining, 12,000 steps | 1,572,864,000 | Sampled validation cross-entropy 1.0910 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/y5ye7t4k) |
| One-epoch 4,096-token RoPE baseline, 4,102 steps | 537,657,344 | Full validation 0.42333 bits per byte; sampled cross-entropy 1.18036 | [W&B](https://wandb.ai/george-nasseem/sprout/runs/i0wis53q) |

The baseline finished on October 7, 2026. Its step-4,102 checkpoint was verified in the private [checkpoint repository](https://huggingface.co/GeorgeCodesSometimes/sprout-checkpoints) at `base/ckpt.pt`; the main checkpoint is at `ckpt.pt`. The 8,192-token and learned-position arms have not run yet. Their cross-entropies should not be compared across tokenizers; the vocabulary comparison uses full-validation bits per byte over the same held-out stories.

## Reproduce

Install `torch numpy regex huggingface_hub wandb`, then run `python data.py`, `python tokenizer.py`, and `python tokenizer.py 8192` to create the seeded split and both tokenizations. The data and checkpoint Hub repositories are currently private; authorized collaborators can instead download the prepared data with:

```bash
hf download GeorgeCodesSometimes/sprout-data --repo-type dataset --local-dir .
mv vocab8192 data/vocab8192
```

On a CUDA machine, run `python model.py` to check the model and its one-batch overfit test. The remaining controlled runs use the same seed, optimizer, and data split as the baseline:

```bash
python train.py --hub GeorgeCodesSometimes/sprout-checkpoints --out checkpoints/vocab8192 --data data/vocab8192 --steps 3974
python train.py --hub GeorgeCodesSometimes/sprout-checkpoints --out checkpoints/learned-pos --pos learned --steps 4102
```

Each arm covers approximately one pass through the same raw stories. The 8,192-token arm needs fewer token steps because its tokenizer compresses the text differently. After both runs, compare full-validation bits per byte and the validation curves, then evaluate the main model and both ablation arms with a fixed, seeded prompt suite.
