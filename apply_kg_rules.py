#!/usr/bin/env python3
"""
HIPE-2026 Post-Processing: KG Hard Rules
=========================================
Take any HIPE-format predictions JSONL and apply temporal hard rules using KG:
  1. If person died BEFORE article date → isAt = FALSE (always)
  2. If person not yet born at article date → at = FALSE, isAt = FALSE
  3. If at = FALSE → isAt = FALSE (consistency)

Use this as a *free* post-processing step on Claude or XLM-R predictions.

Usage:
  python apply_kg_rules.py \\
      --input results/v8_sonnet_de/eval-de.jsonl \\
      --kg_facts kg_facts.jsonl \\
      --output predictions/v8_de_with_rules.jsonl
"""

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hipe_dataset import parse_wd_date, parse_article_date, load_kg_caches


def apply_rules(input_path, output_path, kg_persons, strict_dead=False, verbose=True):
    n_dead = 0
    n_unborn = 0
    n_const = 0
    n_total = 0

    with open(input_path) as f, open(output_path, "w") as fout:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            out_doc = copy.deepcopy(doc)
            article_date = doc.get("date", "")
            art = parse_article_date(article_date)
            for i, pair in enumerate(out_doc.get("sampled_pairs", [])):
                n_total += 1
                pq = pair.get("pers_wikidata_QID") or ""
                pinfo = kg_persons.get(pq, {}) if pq else {}
                death = parse_wd_date(pinfo.get("death_date"))
                birth = parse_wd_date(pinfo.get("birth_date"))

                # Rule 2 (always on): not yet born → at=FALSE, isAt=FALSE
                # Safe rule: someone literally cannot be discussed as present in a place
                # before they were born.
                if birth and art and art < birth:
                    if pair.get("at") != "FALSE" or pair.get("isAt") != "FALSE":
                        n_unborn += 1
                    pair["at"] = "FALSE"
                    pair["isAt"] = "FALSE"
                    continue

                # Rule 1 (OPTIONAL): dead → isAt = FALSE
                # Disabled by default: HIPE annotations contain ambiguous person QIDs
                # (e.g. military rank shared with a famous deceased namesake).
                # Enabling --strict_dead enforces this anyway.
                if strict_dead and death and art and death < art:
                    if pair.get("isAt") == "TRUE":
                        n_dead += 1
                    pair["isAt"] = "FALSE"

                # Rule 3 (consistency): at=FALSE → isAt=FALSE
                if pair.get("at") == "FALSE" and pair.get("isAt") != "FALSE":
                    n_const += 1
                    pair["isAt"] = "FALSE"

            fout.write(json.dumps(out_doc, ensure_ascii=False) + "\n")

    if verbose:
        print(f"  {input_path} → {output_path}")
        print(f"    Total pairs:           {n_total}")
        if strict_dead:
            print(f"    Dead → isAt=FALSE:     {n_dead}")
        print(f"    Unborn → at,isAt=F:    {n_unborn}")
        print(f"    at=F constraint fix:   {n_const}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="Input HIPE-format JSONL files")
    ap.add_argument("--kg_facts", default="kg_facts.jsonl")
    ap.add_argument("--output", help="Output for single input")
    ap.add_argument("--output_dir", help="Output dir, mirroring input filenames")
    ap.add_argument("--strict_dead", action="store_true",
                    help="Apply 'dead → isAt=FALSE' rule (off by default; HIPE QIDs can be ambiguous)")
    args = ap.parse_args()

    print("Loading KG cache...")
    _, kg_persons, _ = load_kg_caches(args.kg_facts)
    print(f"  KG persons: {len(kg_persons)}")
    if args.strict_dead:
        print("  Mode: strict_dead (will force isAt=FALSE for any dead person)")
    else:
        print("  Mode: lenient (only enforces 'unborn' rule)")

    if args.output:
        if len(args.input) != 1:
            raise SystemExit("--output requires exactly one --input")
        apply_rules(args.input[0], args.output, kg_persons, strict_dead=args.strict_dead)
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for in_path in args.input:
            out_path = os.path.join(args.output_dir, os.path.basename(in_path))
            apply_rules(in_path, out_path, kg_persons, strict_dead=args.strict_dead)
    else:
        raise SystemExit("Provide --output or --output_dir")


if __name__ == "__main__":
    main()