# Team Awakened — HIPE-2026 Person–Place Relation Extraction

Code for our submission to the [HIPE-2026](https://hipe-eval.github.io/HIPE-2026/)
shared task on person–place relation extraction from historical documents.

For each (person, place) pair in a document we predict two labels:

- **`at`** — is the person connected to the place? (`TRUE` / `PROBABLE` / `FALSE`)
- **`isAt`** — does that connection hold around the document's date? (`TRUE` / `FALSE`)

We submitted three runs:

| Run | Approach | Model | Params |
|-----|----------|-------|--------|
| 1 | Knowledge-augmented prompting | Claude Sonnet 4 (API) | n/a (closed) |
| 2 | Logic-constrained fine-tuning | XLM-RoBERTa-large | 560,965,127 |
| 3 | Domain-agnostic prompting | Claude Sonnet 4 (API) | n/a (closed) |

A short description of each run is in our CLEF 2026 working notes paper
(citation below).

## Repository layout

```
claude/                     Runs 1 and 3 (prompted Claude Sonnet 4)
  pipeline_v8.py            Run 1 prompt: newspaper-oriented, KG facts, PROBABLE few-shots
  pipeline_v8_literary.py   Run 3 prompt: domain-agnostic (historical/literary)
  apply_kg_rules.py         Post-processing: "unborn -> FALSE" + consistency fix
  submit_test_claude.sh     Driver script for the Claude runs

xlmr/                       Run 2 (fine-tuned encoder)
  hipe_model_fol.py         XLM-R + KG/FOL feature fusion + logic-constrained loss
  hipe_dataset_fol.py       Dataset: 16 Wikidata features + 6 text-pattern features
  hipe_dataset.py           Earlier dataset variant (no FOL text-pattern features)
  train_xlmr_fol.py         Training entry point
  predict_xlmr_fol.py       Inference entry point
  hipe_train_fol_v2.sbatch  SLURM job used to train the submitted checkpoint
  submit_test_xlmr.sbatch   SLURM job for test-set inference

kg/                         Knowledge graph
  kg_reasoner.py            Builds the Wikidata cache and per-pair fact records
  wikidata_enrich.py        Wikidata SPARQL enrichment helpers

artifacts/                  Derived files (for reproducibility)
  wikidata_cache.json       Cached Wikidata facts (persons + locations)
  kg_facts.jsonl            Per-pair fact records
  kg_facts_persons.jsonl    Per-person fact records
  kg_facts_locations.jsonl  Per-location fact records
```

## Setup

```bash
pip install -r requirements.txt
```

The data is **not** included. Download the official HIPE-2026 data from the
[organizers' repository](https://github.com/hipe-eval/HIPE-2026-data) and point
the scripts at it.

For Runs 1 and 3 you need an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=...   # never commit this
```

## Reproducing each run

**Run 1 (Claude, newspaper prompt):**
```bash
python claude/pipeline_v8.py --mode predict \
    --input_file <test-file>.jsonl \
    --wikidata artifacts/wikidata_cache.json \
    --api_key "$ANTHROPIC_API_KEY" \
    --output_dir out_run1
python claude/apply_kg_rules.py \
    --input out_run1/*.jsonl --kg_facts artifacts/kg_facts.jsonl \
    --output_dir out_run1_rules
```

**Run 2 (XLM-R, fine-tuned):**
```bash
# Train (SLURM)
sbatch xlmr/hipe_train_fol_v2.sbatch
# Predict
python xlmr/predict_xlmr_fol.py \
    --checkpoint <run-dir>/best.pt \
    --input <test-file>.jsonl \
    --kg_facts artifacts/kg_facts.jsonl \
    --output out_run2.jsonl
```

**Run 3 (Claude, domain-agnostic prompt):**
```bash
python claude/pipeline_v8_literary.py --mode predict \
    --input_file <test-file>.jsonl \
    --wikidata artifacts/wikidata_cache.json \
    --api_key "$ANTHROPIC_API_KEY" \
    --output_dir out_run3
```

## Citation

<!-- TODO: replace with the final BibTeX once the paper is published. -->
```bibtex
@inproceedings{awakened-hipe2026,
  title     = {Team Awakened at HIPE-2026: Knowledge-Augmented Prompting and
               Logic-Constrained Fine-Tuning for Person--Place Relation Extraction},
  author    = {Vasile, Dragoș-Mitruț and Apostol, Elena-Simona and Truică, Ciprian-Octavian},
  booktitle = {CLEF 2026 Working Notes},
  year      = {2026},
}
```

Please also cite the HIPE-2026 shared task overview papers.

## License

This code is released under the MIT License (see `LICENSE`).