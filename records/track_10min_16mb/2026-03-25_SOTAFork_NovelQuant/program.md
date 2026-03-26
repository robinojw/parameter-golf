# Parameter Golf Autoresearch

## Goal
Minimize val_bpb on FineWeb validation set. Lower is better.
Current SOTA: 1.1194. Target: < 1.114.

## Constraints (HARD — never violate)
- Artifact ≤ 16,000,000 bytes (checked at end of training via "Size check" log line)
- Training completes within MAX_WALLCLOCK_SECONDS=600
- train_gpt.py must stay ≤ 2000 lines
- Do NOT change eval_val(), eval_val_sliding(), or eval_val_ttt_lora()
- Do NOT change the quantization save/load format
- Do NOT change the data loading pipeline
- Do NOT add new pip dependencies
- Do NOT remove any existing features

## How to Run
```bash
SEED=1337 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee run.log
```
Parse result: `grep "pre_quant_exact" run.log` → gives `val_bpb:X.XXXXXXXX`
On 1xH100, expect ~1.40-1.45 BPB (fewer steps than 8xH100). Relative improvements transfer.

## What to Optimize

### Tier 1: Hyperparameters (ENV vars, zero code changes)
These are already in the Hyperparameters class. Change DEFAULTS in the code, not ENV vars.

MATRIX_LR: try [0.020, 0.023, 0.025, 0.027, 0.030]
SCALAR_LR: try [0.020, 0.023, 0.025, 0.027, 0.030]
TIED_EMBED_LR: try [0.030, 0.035, 0.040, 0.045]
WARMDOWN_ITERS: try [3000, 3500, 4000, 4500]
GRAD_CLIP_NORM: try [0.2, 0.3, 0.4, 0.5]
EMA_DECAY: try [0.995, 0.997, 0.998, 0.999]
SWA_EVERY: try [25, 50, 75, 100]
MUON_MOMENTUM: try [0.98, 0.99, 0.995]
BIGRAM_VOCAB_SIZE: try [1024, 1536, 2048, 3072]
XSA_LAST_N: try [0, 2, 3, 4, 5, 6]
ROPE_DIMS: try [0, 8, 16, 24, 32]
VE_DIM: try [64, 96, 128, 192]
VE_LAYERS: try ["9,10", "8,9,10", "7,8,9,10"]
MUON_WD: try [0.02, 0.04, 0.06]
ADAM_WD: try [0.02, 0.04, 0.06]

### Tier 2: Novel features (our additions)
CAUTIOUS_WD: try [0, 1] — cautious weight decay (only decay when sign matches)
ADAPTIVE_QUANT_BITS: try ["", "6,6,6,6,6,6,7,7,7,7,7", "6,6,6,6,7,7,7,7,7,7,7"]

### Tier 3: Small code changes (be careful, test thoroughly)
- Try different QAT start thresholds: LATE_QAT_THRESHOLD in [0.10, 0.15, 0.20, 0.25]
- Try different bigram hash functions (change the constants 36313, 27191)
- Try different init schemes for embeddings (spectral, truncated normal)
- Try different SmearGate init values (currently 0.0, try -1.0 or 1.0)
- Try scaling the skip_weights init (currently ones, try 0.5 or 2.0)

### Tier 4: Architectural experiments (high risk, high reward)
- Try MLP_MULT 2.5 or 3.5 (currently 3.0)
- Try different LeakyReLU slopes (currently 0.5, try 0.3, 0.4, 0.6, 0.7)
- Try adding a second SmearGate after the MLP
- Try different logit_softcap values (currently 30.0, try 20.0, 40.0, 50.0)

## Strategy
1. Start with Tier 1 hyperparameters — sweep one at a time
2. When a Tier 1 change improves val_bpb, commit it and continue
3. After exhausting Tier 1, move to Tier 2
4. Only try Tier 3/4 after Tiers 1-2 are settled
5. Each experiment should change ONE thing. Never change multiple variables at once.

## What Doesn't Work (don't waste experiments on these)
- Curriculum learning on data ordering (tested, negative — FineWeb too homogeneous)
- MTP auxiliary loss (tested, no improvement on this architecture)
- Depth recurrence / weight sharing (compute penalty outweighs depth benefit)
- MoE at 27M scale (negative results, artifact too large)
- int4/int5 quantization for all layers (catastrophic quality loss)
- LoRA-based TTT (no gain over full-weight TTT)
