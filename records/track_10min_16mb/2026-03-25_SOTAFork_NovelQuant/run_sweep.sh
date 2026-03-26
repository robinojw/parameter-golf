#!/bin/bash
set -euo pipefail
cd /workspace/parameter-golf
SCRIPT=records/track_10min_16mb/2026-03-25_SOTAFork_NovelQuant/train_gpt.py
mkdir -p /workspace/logs

run_exp() {
  local name=$1; shift
  echo ">>> START $name at $(date -u +%H:%M:%S)"
  PYTHONUNBUFFERED=1 SEED=1337 EMA_ENABLED=0 TTT_ENABLED=0 "$@" torchrun --standalone --nproc_per_node=1 $SCRIPT 2>&1 | tee /workspace/logs/${name}.txt
  bpb=$(grep "val_bpb" /workspace/logs/${name}.txt | tail -1 | grep -oE "val_bpb:[0-9.]+" | cut -d: -f2)
  echo ">>> RESULT $name val_bpb=$bpb"
  echo "$name $bpb" >> /workspace/logs/results.tsv
}

echo "name val_bpb" > /workspace/logs/results.tsv

run_exp exp1_baseline
run_exp exp2_matrix_lr_030 MATRIX_LR=0.030
run_exp exp3_embed_lr_040 TIED_EMBED_LR=0.040
run_exp exp4_warmdown_4000 WARMDOWN_ITERS=4000
run_exp exp5_cautious_wd CAUTIOUS_WD=1
run_exp exp6_xsa3 XSA_LAST_N=3
run_exp exp7_rope8 ROPE_DIMS=8
run_exp exp8_bigram_1024 BIGRAM_VOCAB_SIZE=1024
run_exp exp9_grad_clip_04 GRAD_CLIP_NORM=0.4

echo "ALL_EXPERIMENTS_DONE"
cat /workspace/logs/results.tsv
