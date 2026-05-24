#!/usr/bin/env bash
#SBATCH --job-name=hipe-r2
#SBATCH --partition=dgxa100,dgxh100,ucsx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/hipe-r2-%j.out
#SBATCH --error=logs/hipe-r2-%j.err

# =============================================================================
# HIPE-2026 Submission — XLM-R pipeline (Run 2: Efficiency)
# =============================================================================
# Runs predict_xlmr.py on test data on GPU node.
#
# Usage on test data drop:
#   sbatch --export=ALL,TEST_DIR=$HOME/HIPE-2026-data/data/test submit_test_xlmr.sbatch
#
# Or with custom OUT_BASE:
#   sbatch --export=ALL,TEST_DIR=...,OUT_BASE=submissions submit_test_xlmr.sbatch
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-synasc2026}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || true
fi

export HF_HOME="${HF_HOME:-$(pwd)/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

TEST_DIR="${TEST_DIR:-$HOME/HIPE-2026-data/data/test}"
OUT_BASE="${OUT_BASE:-submissions}"
KG_FACTS="${KG_FACTS:-kg_facts.jsonl}"
XLMR_CHECKPOINT="${XLMR_CHECKPOINT:-runs/xlmr_fol_v2_low_lambda/best.pt}"
SCHEMA="${SCHEMA:-$HOME/HIPE-2026-data/schemas/hipe-2026-data.schema.json}"
BATCH_SIZE="${BATCH_SIZE:-32}"

RUN2_DIR="$OUT_BASE/run2_efficiency"
mkdir -p "$RUN2_DIR"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] TEST_DIR=$TEST_DIR"
echo "[INFO] OUT_BASE=$OUT_BASE"
echo "[INFO] CHECKPOINT=$XLMR_CHECKPOINT"
nvidia-smi || true

# Detect newspaper vs literary files (XLM-R only on newspapers, not trained on literary)
NEWSPAPER_FILES=()
for f in "$TEST_DIR"/*.jsonl; do
    [ -f "$f" ] || continue
    bn=$(basename "$f")
    case "$bn" in
        *lit*|*work*|*roman*|*novel*) ;;  # skip literary
        *) NEWSPAPER_FILES+=("$f") ;;
    esac
done

echo "[INFO] Found ${#NEWSPAPER_FILES[@]} newspaper files for Run 2"

# ============================================================
# Run 2 — XLM-R predict
# ============================================================
echo "============================================================"
echo "[RUN 2] XLM-R fine-tuned (~560M params)"
echo "============================================================"

if [ ! -f "$XLMR_CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $XLMR_CHECKPOINT"
    exit 1
fi

for f in "${NEWSPAPER_FILES[@]}"; do
    bn=$(basename "$f")
    OUT="$RUN2_DIR/$bn"
    echo ""
    echo "[Run 2] $bn"
    T_START=$(date +%s)
    python3 predict_xlmr_fol.py \
        --checkpoint "$XLMR_CHECKPOINT" \
        --input "$f" \
        --output "$OUT" \
        --kg_facts "$KG_FACTS" \
        --batch_size "$BATCH_SIZE"
    T_END=$(date +%s)
    DURATION=$((T_END - T_START))
    echo "[Run 2 timing] $bn: ${DURATION}s" | tee -a "$OUT_BASE/timing_run2.log"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs -I{} echo "[Run 2 GPU mem] $bn: {} MB" | tee -a "$OUT_BASE/timing_run2.log"
done

echo ""
echo "[Run 2] Done. Files:"
ls -la "$RUN2_DIR"

# ============================================================
# Schema validation
# ============================================================
echo ""
echo "============================================================"
echo "[VALIDATE] Schema check on Run 2"
echo "============================================================"
VALIDATE_LOG="$OUT_BASE/validate_xlmr.log"
> "$VALIDATE_LOG"

if [ -f "$SCHEMA" ]; then
    for f in "$RUN2_DIR"/*.jsonl; do
        [ -f "$f" ] || continue
        # Skip auxiliary probability files - not in HIPE schema
        case "$f" in
            *.probs.jsonl) continue ;;
        esac
        echo "[validate] $f" | tee -a "$VALIDATE_LOG"
        python3 "$HOME/HIPE-2026-data/scripts/check_jsonlschema.py" \
            --schemafile "$SCHEMA" "$f" 2>&1 | tee -a "$VALIDATE_LOG"
    done
fi

echo ""
echo "[OK] Run 2 done. Files in $RUN2_DIR"
