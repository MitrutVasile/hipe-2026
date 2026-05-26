#!/usr/bin/env python3
"""
Inference with trained HipeModel.

Usage:
  # Predict on a single jsonl, write predictions in HIPE official format
  python predict_xlmr.py \
    --checkpoint runs/xlmr_v1/best.pt \
    --input $HOME/HIPE-2026-data/data/sandbox/de-dev.jsonl \
    --output predictions/xlmr_de-dev.jsonl \
    --kg_facts kg_facts.jsonl

  # Predict on multiple files (e.g. all test files at once)
  python predict_xlmr.py \
    --checkpoint runs/xlmr_v1/best.pt \
    --input_glob "$HOME/HIPE-2026-data/data/test/*.jsonl" \
    --output_dir predictions/xlmr_test/ \
    --kg_facts kg_facts.jsonl

Output format: HIPE-2026 official JSONL with `at`, `isAt`, plus optional probabilities
in `at_explanation` / `isAt_explanation` for downstream use.
"""

import argparse
import copy
import json
import os
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from hipe_dataset_fol import (
    HipeDataset, hipe_collate_fn, load_kg_caches,
    AT_LABELS, ISAT_LABELS, KG_FEATURE_DIM,
)
from hipe_model_fol import HipeModelFOL as HipeModel


def load_checkpoint(ckpt_path, device):
    """Load checkpoint. Returns (model, args)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    encoder = saved_args.get("encoder", "FacebookAI/xlm-roberta-large")
    use_kg = not saved_args.get("no_kg", False)
    use_fol = not saved_args.get("no_fol", False)

    from hipe_dataset_fol import FOL_FEATURE_DIM
    model = HipeModel(
        encoder_name=encoder,
        num_at=3, num_isAt=2,
        kg_dim=KG_FEATURE_DIM, fol_dim=FOL_FEATURE_DIM,
        dropout=saved_args.get("dropout", 0.2),
        use_kg=use_kg,
        use_fol=use_fol,
        lambda_hard=saved_args.get("lambda_hard", 0.3),
        lambda_soft=saved_args.get("lambda_soft", 0.1),
    )
    # strict=False because CrossEntropyLoss/FocalLoss may have weight buffers
    # in the state_dict that are training-only (alpha for focal, weight for CE).
    # We don't need them at inference; just ensure encoder + heads load.
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        # Filter out loss-buffer keys (irrelevant at inference)
        critical_missing = [k for k in missing if not k.startswith(("at_loss", "isAt_loss"))]
        if critical_missing:
            print(f"WARNING: critical missing keys: {critical_missing}")
    if unexpected:
        non_loss = [k for k in unexpected if not k.startswith(("at_loss", "isAt_loss"))]
        if non_loss:
            print(f"WARNING: unexpected keys (not loss-related): {non_loss}")
    model.to(device)
    model.eval()
    return model, saved_args


def predict_file(model, tokenizer, device, input_path, output_path,
                 kg_pairs, kg_persons, kg_locations,
                 max_length=320, window=400, batch_size=16,
                 apply_hard_rules=True):
    """Run inference on a HIPE jsonl file, write predictions in official format."""
    ds = HipeDataset([input_path], tokenizer, max_length=max_length,
                     kg_pairs=kg_pairs, kg_persons=kg_persons, kg_locations=kg_locations,
                     has_labels=False, window=window)
    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: hipe_collate_fn(b, pad_token_id=pad_id),
    )

    # Map (doc_id, pair_idx) → predictions
    pred_map = {}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            kg_features = batch["kg_features"].to(device)
            fol_features = batch.get("fol_features")
            if fol_features is not None:
                fol_features = fol_features.to(device)
            pred = model.predict(input_ids, attention_mask, kg_features,
                                 fol_features=fol_features,
                                 apply_hard_rules=apply_hard_rules)
            at_preds = pred["at_preds"].cpu().numpy().tolist()
            isAt_preds = pred["isAt_preds"].cpu().numpy().tolist()
            at_probs = pred["at_probs"].cpu().numpy().tolist()
            isAt_probs = pred["isAt_probs"].cpu().numpy().tolist()
            for did, pi, ap, ip, ap_pr, ip_pr in zip(
                batch["doc_ids"], batch["pair_idxs"], at_preds, isAt_preds, at_probs, isAt_probs
            ):
                pred_map[(did, int(pi))] = {
                    "at": AT_LABELS[ap],
                    "isAt": ISAT_LABELS[ip],
                    "at_probs": ap_pr,
                    "isAt_probs": ip_pr,
                }

    # Read input docs, attach predictions, write output preserving HIPE schema
    docs_out = []
    with open(input_path) as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            out_doc = copy.deepcopy(doc)
            for i, pair in enumerate(out_doc.get("sampled_pairs", [])):
                key = (doc["document_id"], i)
                p = pred_map.get(key)
                if p is None:
                    pair["at"] = pair.get("at") or "FALSE"
                    pair["isAt"] = pair.get("isAt") or "FALSE"
                    pair.setdefault("at_explanation", "")
                    pair.setdefault("isAt_explanation", "")
                else:
                    pair["at"] = p["at"]
                    pair["isAt"] = p["isAt"]
                    # Hard-rule consistency: at=FALSE → isAt=FALSE
                    if pair["at"] == "FALSE":
                        pair["isAt"] = "FALSE"
                    pair["at_explanation"] = ""
                    pair["isAt_explanation"] = ""
            docs_out.append(out_doc)

    # Write
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for d in docs_out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs_out)} docs to {output_path}")

    # Also write probs JSONL for ensemble downstream
    probs_path = output_path.replace(".jsonl", ".probs.jsonl")
    with open(probs_path, "w") as f:
        for (did, pi), p in pred_map.items():
            f.write(json.dumps({
                "document_id": did, "pair_idx": pi,
                "at": p["at"], "isAt": p["isAt"],
                "at_p_FALSE": p["at_probs"][0],
                "at_p_PROBABLE": p["at_probs"][1],
                "at_p_TRUE": p["at_probs"][2],
                "isAt_p_FALSE": p["isAt_probs"][0],
                "isAt_p_TRUE": p["isAt_probs"][1],
            }, ensure_ascii=False) + "\n")
    print(f"Wrote probs to {probs_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", help="Single jsonl file")
    ap.add_argument("--input_glob", help="Glob pattern (e.g. 'data/test/*.jsonl')")
    ap.add_argument("--output", help="Output jsonl path (with --input)")
    ap.add_argument("--output_dir", help="Output directory (with --input_glob)")
    ap.add_argument("--kg_facts", default=None)
    ap.add_argument("--max_length", type=int, default=320)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_hard_rules", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, saved_args = load_checkpoint(args.checkpoint, device)
    encoder = saved_args.get("encoder", "FacebookAI/xlm-roberta-large")
    print(f"Loaded checkpoint {args.checkpoint}, encoder={encoder}")
    tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)

    # KG caches
    kg_pairs, kg_persons, kg_locations = ({}, {}, {})
    if args.kg_facts and os.path.exists(args.kg_facts):
        kg_pairs, kg_persons, kg_locations = load_kg_caches(args.kg_facts)
        print(f"KG: {len(kg_pairs)} pairs, {len(kg_persons)} persons, {len(kg_locations)} locations")

    apply_hard = not args.no_hard_rules
    if args.input:
        if not args.output:
            ap.error("--output required with --input")
        predict_file(model, tokenizer, device, args.input, args.output,
                     kg_pairs, kg_persons, kg_locations,
                     max_length=args.max_length, window=args.window,
                     batch_size=args.batch_size, apply_hard_rules=apply_hard)
    elif args.input_glob:
        if not args.output_dir:
            ap.error("--output_dir required with --input_glob")
        files = sorted(glob(args.input_glob))
        if not files:
            print(f"No files match {args.input_glob}")
            return
        for fpath in files:
            out = os.path.join(args.output_dir, os.path.basename(fpath))
            predict_file(model, tokenizer, device, fpath, out,
                         kg_pairs, kg_persons, kg_locations,
                         max_length=args.max_length, window=args.window,
                         batch_size=args.batch_size, apply_hard_rules=apply_hard)
    else:
        ap.error("--input or --input_glob required")


if __name__ == "__main__":
    main()
