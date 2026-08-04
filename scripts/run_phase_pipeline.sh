#!/bin/bash
# MR-as-Instruction Phase A → B pipeline with comprehensive summary
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llmtrain310
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

PHASE_A_CONFIG="experiments/configs/experiments_config_mrinstr_full_multiseed.json"
PHASE_B_CONFIG="experiments/configs/experiments_config.json"
PHASE_A_OUT="output/experiments/gemma3_4b_mrinstr_full_multiseed_v1"
PHASE_B_OUT="output/experiments/gemma3_4b_metrain_full_v1"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Phase A: wait for finetune, then test ──
log "Phase A: 等待训练完成..."
python3 -c "
import json, time
while True:
    p = json.load(open('$PHASE_A_OUT/_progress.json'))
    finetune_done = sum(1 for v in p.values() if v.get('finetune') == 'completed')
    log_prefix = f'[$(date +%H:%M:%S)] Phase A finetune: {finetune_done}/18'
    print(log_prefix)
    if finetune_done >= 18:
        break
    time.sleep(600)
"
log "Phase A: 全部训练完成，开始评测..."
python scripts/run_batch_experiments.py \
    --config "$PHASE_A_CONFIG" \
    --steps test_original test_mr \
    --seeds 42 43 44 --resume

log "Phase A: 评测完成"

# ── Phase B: full pipeline ──
log "Phase B: 开始全流程 (sample → convert → finetune → test)..."
python scripts/run_batch_experiments.py \
    --config "$PHASE_B_CONFIG" \
    --seeds 42 43 44 --resume

log "Phase B: 完成"

# ── Summary ──
log "生成汇总报告..."
python scripts/summarize_all_results.py

log "全部完成!"
