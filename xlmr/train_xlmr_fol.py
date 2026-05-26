#!/usr/bin/env python3
"""
Train HipeModel on HIPE-2026 data.

Standard fine-tuning loop:
  - AdamW + linear schedule with warmup
  - Mixed precision (fp16) on GPU
  - Class-balanced loss + focal for `at`
  - Per-epoch eval on dev with macro-recall on (at, isAt)
  - Save best checkpoint by GLOBAL = (macro_recall_at + macro_recall_isAt) / 2
  - Save final predictions on dev as JSONL for downstream calibration

Usage:
  python train_xlmr.py \
    --train $HOME/HIPE-2026-data/data/sandbox/de-train.jsonl \
            $HOME/HIPE-2026-data/data/sandbox/en-train.jsonl \
            $HOME/HIPE-2026-data/data/sandbox/fr-train.jsonl \
            $HOME/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de.jsonl \
            $HOME/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-en.jsonl \
            $HOME/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-fr.jsonl \
    --dev   $HOME/HIPE-2026-data/data/sandbox/de-dev.jsonl \
            $HOME/HIPE-2026-data/data/sandbox/en-dev.jsonl \
            $HOME/HIPE-2026-data/data/sandbox/fr-dev.jsonl \
    --kg_facts kg_facts.jsonl \
    --out_dir runs/xlmr_v1 \
    --encoder FacebookAI/xlm-roberta-large \
    --epochs 8 --batch_size 8 --grad_accum 4 --lr 1.5e-5 --seed 42
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from hipe_dataset_fol import (
    HipeDataset, hipe_collate_fn, load_kg_caches,
    AT_LABELS, ISAT_LABELS, KG_FEATURE_DIM,
)
from hipe_model_fol import HipeModelFOL as HipeModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels, num_classes, ignore=-1, power=1.0, max_weight=None):
    """Inverse frequency, normalized to mean=1.

    power: 0.0 = uniform (no class weighting), 0.5 = sqrt-tempered, 1.0 = full inverse-freq
    max_weight: cap the maximum weight (e.g. 2.0 to prevent runaway minorities)
    """
    counts = Counter(l for l in labels if l != ignore)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        w = total / (num_classes * c)
        # Apply power: w^power
        w = w ** power
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()  # normalize to mean=1
    if max_weight is not None:
        weights = np.clip(weights, 1.0 / max_weight, max_weight)
        weights = weights / weights.mean()  # re-normalize
    return torch.tensor(weights)


def macro_recall(preds, labels, num_classes, ignore=-1):
    """Recall per class + macro avg."""
    per_class = []
    for c in range(num_classes):
        mask = (np.array(labels) == c)
        if mask.sum() == 0:
            per_class.append(0.0)
            continue
        correct = (np.array(preds)[mask] == c).sum()
        per_class.append(correct / mask.sum())
    return float(np.mean(per_class)), per_class


def confusion_matrix(preds, labels, num_classes):
    """[gold][pred] count matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        if l >= 0 and p >= 0:
            cm[l, p] += 1
    return cm


def evaluate(model, loader, device, apply_hard_rules=True):
    """Run evaluation, return metrics dict + per-example info."""
    model.eval()
    all_at_preds, all_at_labels = [], []
    all_isAt_preds, all_isAt_labels = [], []
    all_at_probs = []
    all_isAt_probs = []
    all_doc_ids, all_pair_idxs, all_languages = [], [], []

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
            all_at_preds.extend(pred["at_preds"].cpu().numpy().tolist())
            all_isAt_preds.extend(pred["isAt_preds"].cpu().numpy().tolist())
            all_at_probs.extend(pred["at_probs"].cpu().numpy().tolist())
            all_isAt_probs.extend(pred["isAt_probs"].cpu().numpy().tolist())
            all_at_labels.extend(batch["at_labels"].cpu().numpy().tolist())
            all_isAt_labels.extend(batch["isAt_labels"].cpu().numpy().tolist())
            all_doc_ids.extend(batch["doc_ids"])
            all_pair_idxs.extend(batch["pair_idxs"])
            all_languages.extend(batch["languages"])

    at_macro, at_per = macro_recall(all_at_preds, all_at_labels, num_classes=3)
    is_macro, is_per = macro_recall(all_isAt_preds, all_isAt_labels, num_classes=2)
    cm_at = confusion_matrix(all_at_preds, all_at_labels, 3)
    cm_is = confusion_matrix(all_isAt_preds, all_isAt_labels, 2)

    metrics = {
        "at_macro_recall": at_macro,
        "isAt_macro_recall": is_macro,
        "global": (at_macro + is_macro) / 2.0,
        "at_per_label": dict(zip(AT_LABELS, at_per)),
        "isAt_per_label": dict(zip(ISAT_LABELS, is_per)),
        "at_confusion": cm_at.tolist(),
        "isAt_confusion": cm_is.tolist(),
    }

    # Per-language breakdown
    by_lang = {}
    for lang in set(all_languages):
        if lang == "?": continue
        idxs = [i for i, l in enumerate(all_languages) if l == lang]
        if not idxs: continue
        a_p = [all_at_preds[i] for i in idxs]
        a_l = [all_at_labels[i] for i in idxs]
        i_p = [all_isAt_preds[i] for i in idxs]
        i_l = [all_isAt_labels[i] for i in idxs]
        a_m, _ = macro_recall(a_p, a_l, 3)
        i_m, _ = macro_recall(i_p, i_l, 2)
        by_lang[lang] = {"at": a_m, "isAt": i_m, "global": (a_m + i_m) / 2.0, "n": len(idxs)}
    metrics["by_language"] = by_lang

    pred_records = [
        {
            "document_id": did, "pair_idx": pi, "language": lang,
            "at_pred": AT_LABELS[ap], "isAt_pred": ISAT_LABELS[ip],
            "at_label": AT_LABELS[al] if al >= 0 else None,
            "isAt_label": ISAT_LABELS[il] if il >= 0 else None,
            "at_probs": ap_probs,
            "isAt_probs": ip_probs,
        }
        for did, pi, lang, ap, ip, al, il, ap_probs, ip_probs in zip(
            all_doc_ids, all_pair_idxs, all_languages,
            all_at_preds, all_isAt_preds, all_at_labels, all_isAt_labels,
            all_at_probs, all_isAt_probs,
        )
    ]
    return metrics, pred_records


def print_metrics(epoch, metrics, prefix=""):
    print(f"\n{prefix}Epoch {epoch}: GLOBAL={metrics['global']:.4f}  "
          f"at_macro={metrics['at_macro_recall']:.4f}  isAt_macro={metrics['isAt_macro_recall']:.4f}")
    print(f"  at per-label: {metrics['at_per_label']}")
    print(f"  isAt per-label: {metrics['isAt_per_label']}")
    print(f"  at confusion (rows=gold, cols=pred [F,P,T]):")
    for i, row in enumerate(metrics["at_confusion"]):
        print(f"    gold={AT_LABELS[i]:9s}: {row}")
    print(f"  isAt confusion (rows=gold, cols=pred [F,T]):")
    for i, row in enumerate(metrics["isAt_confusion"]):
        print(f"    gold={ISAT_LABELS[i]:9s}: {row}")
    if "by_language" in metrics and metrics["by_language"]:
        print(f"  Per-language:")
        for lang, m in sorted(metrics["by_language"].items()):
            print(f"    {lang}: GLOBAL={m['global']:.4f}  at={m['at']:.4f}  isAt={m['isAt']:.4f}  (n={m['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--kg_facts", default=None, help="Path to kg_facts.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder", default="FacebookAI/xlm-roberta-large")
    ap.add_argument("--max_length", type=int, default=320)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.08)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_kg", action="store_true", help="Disable KG features (ablation)")
    ap.add_argument("--no_fol", action="store_true", help="Disable FOL text features (ablation)")
    ap.add_argument("--lambda_hard", type=float, default=0.3,
                    help="Weight for hard logic constraints in loss")
    ap.add_argument("--lambda_soft", type=float, default=0.1,
                    help="Weight for soft FOL constraints in loss")
    ap.add_argument("--no_focal", action="store_true", help="Disable focal loss for `at`")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss gamma (1.0 mild, 2.0 default, 3.0 aggressive)")
    ap.add_argument("--cw_power", type=float, default=1.0,
                    help="Class weight power: 0=uniform, 0.5=sqrt-tempered, 1.0=full inv-freq")
    ap.add_argument("--cw_max", type=float, default=None,
                    help="Cap class weight max (e.g. 2.0). None=no cap.")
    ap.add_argument("--no_hard_rules", action="store_true", help="Disable post-hoc hard rules at eval")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"=== Config ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # KG caches
    kg_pairs, kg_persons, kg_locations = ({}, {}, {})
    if args.kg_facts and os.path.exists(args.kg_facts):
        kg_pairs, kg_persons, kg_locations = load_kg_caches(args.kg_facts)
        print(f"KG: {len(kg_pairs)} pairs, {len(kg_persons)} persons, {len(kg_locations)} locations")
    else:
        print(f"WARN: no KG cache at {args.kg_facts}; KG features will be all zeros")

    # Tokenizer + datasets
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    print(f"Loading train data...")
    train_ds = HipeDataset(args.train, tokenizer, max_length=args.max_length,
                           kg_pairs=kg_pairs, kg_persons=kg_persons, kg_locations=kg_locations,
                           has_labels=True, window=args.window)
    print(f"  train: {len(train_ds)} examples")
    print(f"Loading dev data...")
    dev_ds = HipeDataset(args.dev, tokenizer, max_length=args.max_length,
                         kg_pairs=kg_pairs, kg_persons=kg_persons, kg_locations=kg_locations,
                         has_labels=True, window=args.window)
    print(f"  dev: {len(dev_ds)} examples")

    # Class weights from train
    at_labels_train = [ex["at_label"] for ex in train_ds.examples]
    isAt_labels_train = [ex["isAt_label"] for ex in train_ds.examples]
    at_weights = compute_class_weights(at_labels_train, num_classes=3,
                                        power=args.cw_power, max_weight=args.cw_max)
    isAt_weights = compute_class_weights(isAt_labels_train, num_classes=2,
                                          power=args.cw_power, max_weight=args.cw_max)
    print(f"at class weights: {at_weights.tolist()}  ({AT_LABELS})")
    print(f"isAt class weights: {isAt_weights.tolist()}  ({ISAT_LABELS})")

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: hipe_collate_fn(b, pad_token_id=pad_id),
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: hipe_collate_fn(b, pad_token_id=pad_id),
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    from hipe_dataset_fol import FOL_FEATURE_DIM
    model = HipeModel(
        encoder_name=args.encoder,
        num_at=3, num_isAt=2,
        kg_dim=KG_FEATURE_DIM, fol_dim=FOL_FEATURE_DIM,
        dropout=args.dropout,
        use_kg=(not args.no_kg),
        use_fol=(not args.no_fol),
        at_class_weights=at_weights,
        isAt_class_weights=isAt_weights,
        use_focal_at=(not args.no_focal),
        focal_gamma=args.focal_gamma,
        lambda_hard=args.lambda_hard,
        lambda_soft=args.lambda_soft,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Optimizer + scheduler
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.grad_accum) * int(args.epochs)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    scaler = GradScaler() if torch.cuda.is_available() else None
    print(f"Total opt steps: {total_steps}, warmup: {warmup_steps}")

    # Train
    best_global = -1.0
    best_epoch = -1
    log = []
    apply_hard = (not args.no_hard_rules)

    for epoch in range(int(args.epochs)):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        optim.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            kg_features = batch["kg_features"].to(device, non_blocking=True)
            fol_features = batch.get("fol_features")
            if fol_features is not None:
                fol_features = fol_features.to(device, non_blocking=True)
            at_labels = batch["at_labels"].to(device, non_blocking=True)
            isAt_labels = batch["isAt_labels"].to(device, non_blocking=True)

            if scaler is not None:
                with autocast(dtype=torch.float16):
                    out = model(input_ids, attention_mask, kg_features, fol_features,
                                at_labels=at_labels, isAt_labels=isAt_labels)
                    loss = out["loss"] / args.grad_accum
                scaler.scale(loss).backward()
            else:
                out = model(input_ids, attention_mask, kg_features, fol_features,
                            at_labels=at_labels, isAt_labels=isAt_labels)
                loss = out["loss"] / args.grad_accum
                loss.backward()

            running_loss += out["loss"].item()
            n_batches += 1

            if (step + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                scheduler.step()
                optim.zero_grad()

            if (step + 1) % 50 == 0:
                avg = running_loss / n_batches
                lr_now = optim.param_groups[0]["lr"]
                print(f"  epoch {epoch} step {step+1}/{len(train_loader)} loss={avg:.4f} lr={lr_now:.2e}")

        # Eval
        avg_loss = running_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        metrics, pred_records = evaluate(model, dev_loader, device, apply_hard_rules=apply_hard)
        print_metrics(epoch, metrics, prefix=f"[Eval ({elapsed:.0f}s, train_loss={avg_loss:.4f})] ")

        log.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "elapsed": elapsed,
            **{k: v for k, v in metrics.items() if k != "by_language"},
            "by_language": metrics["by_language"],
        })

        # Save best
        if metrics["global"] > best_global:
            best_global = metrics["global"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "tokenizer_name": args.encoder,
                "epoch": epoch,
                "metrics": metrics,
            }, out_dir / "best.pt")
            with (out_dir / "best_dev_predictions.jsonl").open("w") as f:
                for r in pred_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with (out_dir / "best_metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  ** new best: GLOBAL={best_global:.4f} (epoch {epoch}) **")

    # Final: also save last model
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "tokenizer_name": args.encoder,
        "epoch": epoch,
    }, out_dir / "last.pt")

    with (out_dir / "train_log.json").open("w") as f:
        json.dump(log, f, indent=2)

    print(f"\n=== Training done ===")
    print(f"Best GLOBAL: {best_global:.4f} at epoch {best_epoch}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
