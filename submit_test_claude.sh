#!/usr/bin/env bash
# =============================================================================
# HIPE-2026 Submission — Claude pipeline (Run 1 + Run 3)
# =============================================================================
# Runs Claude-based pipelines on the login node (no GPU needed):
#   Run 1 — Accuracy:        Claude v8 + lenient KG rules
#   Run 3 — Generalization:  Claude v8_literary
#
# Run 2 (XLM-R) is in a separate sbatch — submit_test_xlmr.sbatch
# Both can run in parallel.
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   export TEST_DIR=$HOME/HIPE-2026-data/data/test
#   bash submit_test_claude.sh
# =============================================================================

set -euo pipefail

TEST_DIR="${TEST_DIR:-$HOME/HIPE-2026-data/data/test}"
OUT_BASE="${OUT_BASE:-submissions}"
KG_FACTS="${KG_FACTS:-kg_facts.jsonl}"
WIKIDATA_CACHE="${WIKIDATA_CACHE:-wikidata_cache.json}"
SCHEMA="${SCHEMA:-$HOME/HIPE-2026-data/schemas/hipe-2026-data.schema.json}"

RUN1_DIR="$OUT_BASE/run1_accuracy"
RUN3_DIR="$OUT_BASE/run3_generalization"

echo "[INFO] HIPE-2026 Submission — Claude (Run 1 + Run 3)"
echo "[INFO] $(date)"
echo "[INFO] TEST_DIR=$TEST_DIR"
echo "[INFO] OUT_BASE=$OUT_BASE"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "[ERROR] ANTHROPIC_API_KEY not set"; exit 1
fi
if [ ! -d "$TEST_DIR" ]; then
    echo "[ERROR] Test dir not found: $TEST_DIR"; exit 1
fi

mkdir -p "$RUN1_DIR" "$RUN3_DIR"

# Detect newspaper vs literary files
NEWSPAPER_FILES=()
LITERARY_FILES=()
for f in "$TEST_DIR"/*.jsonl; do
    [ -f "$f" ] || continue
    bn=$(basename "$f")
    case "$bn" in
        *lit*|*work*|*roman*|*novel*) LITERARY_FILES+=("$f") ;;
        *) NEWSPAPER_FILES+=("$f") ;;
    esac
done

echo "[INFO] Newspapers: ${#NEWSPAPER_FILES[@]} | Literary: ${#LITERARY_FILES[@]}"

# ============================================================
# Run 1 — Accuracy: Claude v8 + lenient KG rules
# ============================================================
echo ""
echo "============================================================"
echo "[RUN 1] Accuracy: Claude v8 + lenient KG rules"
echo "============================================================"
TMP_R1=$(mktemp -d)
mkdir -p "$TMP_R1/preds_raw"

for f in "${NEWSPAPER_FILES[@]}"; do
    bn=$(basename "$f")
    lang=$(head -1 "$f" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('language','?'))" 2>/dev/null || echo "?")
    if [ "$lang" = "?" ]; then
        lang=$(echo "$bn" | grep -oE '^[a-z]{2}' || echo "")
    fi
    echo "[Run 1] $bn  (lang=$lang)"

    SINGLE_DIR=$(mktemp -d)
    cp "$f" "$SINGLE_DIR/"

    # Run pipeline_v8 (the proven prompt) on this file
    T_START=$(date +%s)
    python3 pipeline_v8.py --mode predict \
        --data_dir "$SINGLE_DIR" \
        --api_key "$ANTHROPIC_API_KEY" \
        --wikidata "$WIKIDATA_CACHE" \
        --lang "$lang" \
        --output_dir "$TMP_R1/preds_raw"
    T_END=$(date +%s)
    DURATION=$((T_END - T_START))
    echo "[Run 1 timing] $bn (lang=$lang): ${DURATION}s" | tee -a "$OUT_BASE/timing_run1.log"

    rm -rf "$SINGLE_DIR"
done

echo "[Run 1] Applying lenient KG rules..."
python3 apply_kg_rules.py \
    --input "$TMP_R1/preds_raw"/*.jsonl \
    --kg_facts "$KG_FACTS" \
    --output_dir "$RUN1_DIR"

echo "[Run 1] Done."
ls -la "$RUN1_DIR"

# ============================================================
# Run 3 — Generalization: Claude v8_literary
# ============================================================
echo ""
echo "============================================================"
echo "[RUN 3] Generalization: Claude v8_literary"
echo "============================================================"

ALL_FILES=("${NEWSPAPER_FILES[@]}" "${LITERARY_FILES[@]}")
for f in "${ALL_FILES[@]}"; do
    bn=$(basename "$f")
    lang=$(head -1 "$f" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('language','?'))" 2>/dev/null || echo "?")
    if [ "$lang" = "?" ]; then
        lang=$(echo "$bn" | grep -oE '^[a-z]{2}' || echo "")
    fi
    echo "[Run 3] $bn  (lang=$lang)"

    SINGLE_DIR=$(mktemp -d)
    cp "$f" "$SINGLE_DIR/"

    T_START=$(date +%s)
    python3 pipeline_v8_literary.py --mode predict \
        --data_dir "$SINGLE_DIR" \
        --api_key "$ANTHROPIC_API_KEY" \
        --wikidata "$WIKIDATA_CACHE" \
        --lang "$lang" \
        --output_dir "$RUN3_DIR"
    T_END=$(date +%s)
    DURATION=$((T_END - T_START))
    echo "[Run 3 timing] $bn (lang=$lang): ${DURATION}s" | tee -a "$OUT_BASE/timing_run3.log"

    rm -rf "$SINGLE_DIR"
done

echo "[Run 3] Done."
ls -la "$RUN3_DIR"

# ============================================================
# Schema validation
# ============================================================
echo ""
echo "============================================================"
echo "[VALIDATE] Schema check on Run 1 + Run 3"
echo "============================================================"
VALIDATE_LOG="$OUT_BASE/validate_claude.log"
> "$VALIDATE_LOG"

if [ -f "$SCHEMA" ]; then
    for d in "$RUN1_DIR" "$RUN3_DIR"; do
        for f in "$d"/*.jsonl; do
            [ -f "$f" ] || continue
            echo "[validate] $f" | tee -a "$VALIDATE_LOG"
            python3 "$HOME/HIPE-2026-data/scripts/check_jsonlschema.py" \
                --schemafile "$SCHEMA" "$f" 2>&1 | tee -a "$VALIDATE_LOG"
        done
    done
else
    echo "[WARN] Schema not found, skipping validation"
fi

echo ""
echo "============================================================"
echo "[DONE] Claude pipelines complete (Run 1 + Run 3)"
echo "============================================================"
echo "  Run 1: $RUN1_DIR"
echo "  Run 3: $RUN3_DIR"
echo "  Validation: $VALIDATE_LOG"
echo ""
echo "Don't forget Run 2! Submit XLM-R sbatch:"
echo "  sbatch submit_test_xlmr.sbatch"