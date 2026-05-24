#!/usr/bin/env bash
#SBATCH --job-name=hipe-fol-v2
#SBATCH --partition=ucsx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/hipe-fol-v2-%j.out
#SBATCH --error=logs/hipe-fol-v2-%j.err

# =============================================================================
# Variant v2: Same FOL features but MUCH lower constraint weights
# Hypothesis: λ_hard=0.3 was too aggressive, dominated CE loss
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export HF_HOME="${HF_HOME:-$(pwd)/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

OUT_DIR="${OUT_DIR:-runs/xlmr_fol_v2_low_lambda}"
mkdir -p "$OUT_DIR"

echo "[INFO] FOL v2: low lambdas (0.05/0.02)"
nvidia-smi || true

python3 train_xlmr_fol.py \
    --train $HOME/HIPE-2026-data/data/sandbox/de-train.jsonl \
            $HOME/HIPE-2026-data/data/sandbox/en-train.jsonl \
            $HOME/HIPE-2026-data/data/sandbox/fr-train.jsonl \
    --dev   $HOME/HIPE-2026-data/data/sandbox/de-dev.jsonl \
            $HOME/HIPE-2026-data/data/sandbox/en-dev.jsonl \
            $HOME/HIPE-2026-data/data/sandbox/fr-dev.jsonl \
    --kg_facts kg_facts.jsonl \
    --out_dir "$OUT_DIR" \
    --encoder FacebookAI/xlm-roberta-large \
    --epochs 20 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 1.5e-5 \
    --cw_power 0.3 \
    --focal_gamma 1.0 \
    --max_length 320 \
    --lambda_hard 0.05 \
    --lambda_soft 0.02 \
    --seed 42

echo "[OK] Done. Best metrics:"
cat "$OUT_DIR/best_metrics.json" 2>/dev/null | python3 -m json.tool || echo "no metrics"
