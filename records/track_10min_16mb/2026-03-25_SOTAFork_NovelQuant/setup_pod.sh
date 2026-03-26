#!/bin/bash
set -euo pipefail

# Parameter Golf: RunPod 1xH100 setup for autoresearch
# Run this script on a fresh RunPod pod to set up everything needed.

echo "=== Parameter Golf Autoresearch Setup ==="
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# 1. Install deps
echo ">>> Installing dependencies..."
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>&1 | tail -1
pip install zstandard 2>&1 | tail -1
echo "DEPS_OK"

# 2. Clone repo
echo ">>> Cloning repository..."
cd /workspace
rm -rf parameter-golf
git clone -b submission/sota-fork-novel https://github.com/robinojw/parameter-golf.git
cd parameter-golf
echo "CLONE_OK"

# 3. Download data
echo ">>> Downloading FineWeb data..."
python3 data/cached_challenge_fineweb.py --variant sp1024
echo "DATA_OK"

# 4. Verify GPU
echo ">>> GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}')"

# 5. Quick smoke test (single step, verify script runs)
echo ">>> Smoke test..."
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-25_SOTAFork_NovelQuant
PYTHONUNBUFFERED=1 SEED=1337 MAX_WALLCLOCK_SECONDS=30 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tail -5
echo "SMOKE_OK"

# 6. Setup autoresearch working directory
echo ">>> Setting up autoresearch working dir..."
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-25_SOTAFork_NovelQuant
git config user.email "autoresearch@parameter-golf"
git config user.name "autoresearch"

# 7. Install Claude Code
echo ">>> Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code 2>&1 | tail -1

echo ""
echo "=== SETUP COMPLETE ==="
echo ""
echo "Next steps:"
echo "  1. Set your Anthropic API key:"
echo "     export ANTHROPIC_API_KEY='sk-ant-...'"
echo ""
echo "  2. Start autoresearch:"
echo "     cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-25_SOTAFork_NovelQuant"
echo "     claude --dangerously-skip-permissions"
echo ""
echo "  3. In Claude Code, say:"
echo '     "Read program.md. Run experiments to minimize val_bpb.'
echo '      After each run, if pre_quant val_bpb improved, commit the change.'
echo '      If it got worse, revert with git checkout -- train_gpt.py.'
echo '      Keep going until you exhaust all Tier 1 hyperparameters."'
echo ""
echo "  4. Monitor: tail -f run.log (in another terminal)"
echo ""
echo "Pod is burning \$2.69/hr. Don't forget to stop it when done!"
