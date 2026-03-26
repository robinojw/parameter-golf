# Record: Per-Row GPTQ-lite + Cautious Weight Decay + Tuned Hyperparameters (mean val_bpb=1.1447, 3 seeds)

## Results

| Seed | val_bpb (post-quant) |
|------|---------------------|
| 1337 | 1.14485 |
| 42 | 1.14481 |
| 2025 | 1.14441 |
| **Mean** | **1.14469 ± 0.00025** |

Competition SOTA ([PR #549](https://github.com/openai/parameter-golf/pull/549)): 1.1194

## Base

I forked the current SOTA script from
[2026-03-23_LeakyReLU_LegalTTT_ParallelMuon](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon)
(PR #549 by @abaybektursun), which provides the full proven stack: Parameter Banking
with Parallel Muon, LeakyReLU(0.5)², XSA on last 4 layers, Partial RoPE (16/64 dims),
Layerwise LN Scale, VE128, Flash Attention 3, EMA + SWA, SmearGate, Legal Score-First
TTT, and int6 quantization with lzma compression.

## My Changes

### 1. Per-Row GPTQ-lite Quantization

The SOTA's `quantize_int6_per_row` searches 5 clipping percentiles but picks a single
global winner per tensor (the percentile with lowest mean-squared error across all rows).
I changed this to search **per row** — each row of each weight matrix independently picks
its own optimal clipping threshold from [0.995, 0.999, 0.9995, 0.9999, 1.0] based on
that row's specific MSE. This is more expensive (requires `.sort()` per tensor) but runs
only once post-training.

Different rows have different outlier profiles. Attention projection rows tend to have
sharp outliers (benefit from aggressive clipping at 0.995), while MLP rows have flatter
distributions (benefit from no clipping at 1.0). Per-row selection adapts to each.

### 2. Cautious Weight Decay (from modded-nanogpt PR #154)

Standard weight decay unconditionally shrinks all weights by `lr * wd` every step. Cautious
weight decay only applies the decay when `sign(gradient) == sign(weight)` — i.e., when the
gradient is already pushing the weight toward zero. When the gradient opposes the weight's
sign, decay is suppressed to avoid fighting the training signal.

I implemented this in the Muon optimizer's `step()` method. The flag is controlled by
`CAUTIOUS_WD` (default: enabled). This technique was proven in the modded-nanogpt
competition but has not appeared on the parameter-golf leaderboard.

### 3. Hyperparameter Tuning via Autoresearch-Style Sweep

I ran a systematic 9-experiment sweep on 1xH100, testing one hyperparameter change at a
time against the SOTA defaults. Each experiment trained for 600s and I compared val_bpb at
step 500 (the first evaluation checkpoint). The sweep found 3 winning defaults:

| Parameter | SOTA Default | My Default | BPB Delta at Step 500 |
|-----------|-------------|------------|----------------------|
| BIGRAM_VOCAB_SIZE | 2048 | 1024 | -0.007 (best single change) |
| TIED_EMBED_LR | 0.035 | 0.040 | -0.004 |
| GRAD_CLIP_NORM | 0.3 | 0.4 | -0.001 |
| CAUTIOUS_WD | 0 (N/A) | 1 | -0.002 |

Changes that hurt: MATRIX_LR=0.030 (+0.044), WARMDOWN_ITERS=4000 (+0.015), ROPE_DIMS=8
(+0.024). These were reverted.

The smaller BigramHash (1024 vs 2048) was the biggest surprise. The larger table consumed
artifact bytes without capturing proportionally more useful bigram patterns for the 1024-token
vocabulary.

## Run Command

```bash
PYTHONUNBUFFERED=1 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Environment

- 8x NVIDIA H100 80GB HBM3 (SXM)
- runpod/parameter-golf:latest
- torch 2.9.1+cu128, CUDA 12.8
- flash_attn_3 (pre-built wheel), zstandard
