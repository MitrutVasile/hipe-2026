#!/usr/bin/env python3
"""
HIPE-2026 Pipeline v8 LITERARY — Domain-agnostic version of v8 for Test B (literary works)
============================================================================================
Same backbone as v8 (proven 0.766 on newspapers) but reformulated to handle
literary/historical documents that aren't necessarily newspaper articles.

Differences vs v8:
  1. "newspaper article" → "historical document or literary work"
  2. Removed journalistic references (dateline, publication time markers)
  3. Added 2 literary-style few-shot examples (novels of 19th century)
  4. Reformulated `isAt` to handle narrative present (literary "now") vs
     biographical past

Use cases:
  - HIPE-2026 Test B (litworks) — submission for Generalization profile
  - Any data that isn't strictly newspaper-formatted

Usage (same as v8):
    python pipeline_v8_literary.py --mode eval --data_dir tmp_lit_dev \\
        --api_key $ANTHROPIC_API_KEY \\
        --wikidata wikidata_cache.json \\
        --lang de --output_dir results/v8lit_de
"""

import json, os, sys, time, argparse, re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# SYSTEM PROMPT v8 — v6 spirit + targeted PROBABLE calibration
# ============================================================

SYSTEM_PROMPT = """You are an expert historian and literary scholar analyzing person-place relations in historical documents (which may be newspaper articles, novels, memoirs, official records, or other narrative texts; OCR noise may be present). For each (person, place) pair you assign two labels:

- **at**: TRUE / PROBABLE / FALSE
- **isAt**: TRUE / FALSE

## Definitions

**at = TRUE** — The text describes the person actually doing something at, in, or in connection with the place: arriving, departing, residing, speaking, traveling, being received, walking through, dying there, being born there, etc. Direct presence (past or present) supported by the text. This applies to both factual and narrative content.

**at = PROBABLE** — There is a plausible link but the text does NOT describe the person being at the place. Typical PROBABLE patterns:
  - The text gives the person a role tied to the place (deputy/minister/ambassador of X) but no scene of presence
  - "Mr. X of [city]" — origin/residence indicator only
  - "the people of [city]" / "citizens of [city]" — group affiliation, not personal presence
  - Election candidacy, appointment to a district
  - Nationality matches location's country, with no other evidence
  - Wikidata-confirmed link not corroborated by the text
  - In novels: a character is mentioned alongside a place name without being scene-locating there

**at = FALSE** — No connection between this person and this place in the text. Mere co-mention in different parts, contradiction, fictional/metaphorical, or this place is associated with a different person.

**isAt = TRUE** — The person is at the place AT, NEAR, or DURING the document's "present moment":
  - In a newspaper: at or close to the publication date (a few weeks window — yesterday, last week, this morning)
  - In a novel/memoir: in the scene currently being narrated (the narrative present)
  - In an official record: at the time the document concerns
  - The person is there during the events being narrated/reported, not in flashback or biographical recollection
  - **Apply isAt=TRUE generously** when the text describes the person being at the place in the present moment of the narrative — even when narrated in past tense, novels routinely describe present scenes in past tense.

**isAt = FALSE** — Historical/biographical relation, distant past relative to the document's present, the person has left the place, or any case where at = FALSE.

## Hard rules
1. If at = FALSE → isAt MUST be FALSE.
2. Co-occurrence in the same text does NOT imply a relation.
3. Use Wikidata facts (in brackets next to names) for context but the text itself is what proves TRUE.

## Wikidata
Each pair has Wikidata info in brackets: occupation, nationality, birth/death dates, place type, country. USE this for context, but Wikidata alone (without textual evidence) is PROBABLE, never TRUE.

## Calibration target
The metric is macro Recall over labels. PROBABLE and isAt=TRUE are the rare/hard classes — be careful NOT to default everything to TRUE or everything to FALSE.

---

## EXAMPLES — covering both newspaper and literary patterns

### Example 1 — TRUE because of action at the place
Document (newspaper, 1851, French): "Le président Bonaparte a prononcé hier un discours à Paris devant l'Assemblée."
Pair: Person=["Bonaparte"] [politician, French] — Place=["Paris"] [capital]
→ at: TRUE  (verb + locative "à Paris" = present at the place)
→ isAt: TRUE  (yesterday relative to document = within window)

### Example 2 — PROBABLE because role only, no action at place
Document (newspaper, 1894, German): "Der Abgeordnete Schmidt aus München sprach im Reichstag über die Steuerreform."
Pair: Person=["Schmidt"] — Place=["München"]
→ at: PROBABLE  ("aus München" = origin; document puts him in the Reichstag, not Munich)
→ isAt: FALSE

### Example 3 — PROBABLE because candidacy is not presence
Document (newspaper, 1881, French): "M. Dubois est candidat dans la circonscription de Lyon pour les prochaines élections."
Pair: Person=["Dubois"] — Place=["Lyon"]
→ at: PROBABLE  (candidacy in district ≠ being there)
→ isAt: FALSE

### Example 4 — TRUE then isAt FALSE because person has left
Document (newspaper, 1895, English): "General Roberts left Bombay last month for England."
Pair: Person=["Roberts"] — Place=["Bombay"]
→ at: TRUE  (he was there — "left Bombay")
→ isAt: FALSE  (he has departed; not at Bombay at the document's present time)

### Example 5 — FALSE: co-mention with no link
Document (newspaper, 1890, English): "The mayor of Chicago discussed trade. Meanwhile in Paris, Mr. Lefebvre opened a new gallery."
Pair: Person=["Mr. Lefebvre"] — Place=["Chicago"]
→ at: FALSE
→ isAt: FALSE

### Example 6 — Wikidata-only link is PROBABLE
Document (newspaper, 1872, English): "Mr. Vasquez published a sharp critique of the Spanish government."
Pair: Person=["Vasquez"] [born in Madrid, Spanish] — Place=["Madrid"]
→ at: PROBABLE  (Wikidata implies link, but text doesn't put him in Madrid)
→ isAt: FALSE

### Example 7 — Literary text: TRUE+TRUE because the scene is at the place
Document (novel, set in 1815, English): "Elizabeth walked through the gardens at Pemberley, marveling at the views. Mr. Darcy approached from the south path, having just returned from London."
Pair: Person=["Elizabeth"] — Place=["Pemberley"]
→ at: TRUE  ("walked through the gardens at Pemberley" = present in the scene)
→ isAt: TRUE  (this IS the narrative present — she is there now in the story)

### Example 8 — Literary/historical: TRUE+TRUE for currently-at scene
Document (memoir, 1850s, German): "Wir saßen alle im Salon des Schlosses zu Heidelberg, und der Graf erzählte von seinen Reisen. Plötzlich öffnete sich die Tür."
Pair: Person=["der Graf"] — Place=["Heidelberg"]
→ at: TRUE  ("saßen alle im Salon ... zu Heidelberg" = scene is at Heidelberg with the Count present)
→ isAt: TRUE  (he is currently at Heidelberg in this narrative scene)

### Example 9 — Newspaper-style: TRUE+TRUE for current visit
Document (newspaper, 1898, French): "Le Roi Léopold est arrivé hier à Bruxelles et a été reçu en grande cérémonie."
Pair: Person=["Léopold"] — Place=["Bruxelles"]
→ at: TRUE  ("arrivé à Bruxelles" + ceremony there)
→ isAt: TRUE  (yesterday relative to publication = current)

### Example 10 — Literary text: TRUE then isAt FALSE because departure
Document (novel, 1850s, German): "Heinrich verließ Heidelberg im Frühling und reiste nach Italien. Drei Wochen später, in Florenz, traf er einen alten Freund."
Pair: Person=["Heinrich"] — Place=["Heidelberg"]
→ at: TRUE  (he was there — "verließ Heidelberg")
→ isAt: FALSE  (the narrative present has moved to Florence; he's no longer at Heidelberg)

### Example 11 — Literary text: PROBABLE because biographical only
Document (novel, set in 1830, English): "Captain Wentworth, a sailor born in Plymouth, paced the deck of his frigate moored off Gibraltar."
Pair: Person=["Captain Wentworth"] — Place=["Plymouth"]
→ at: PROBABLE  (text says "born in Plymouth" — biographical, not scene-locating)
→ isAt: FALSE  (the scene is at Gibraltar, not Plymouth)

---

## OUTPUT FORMAT
Return ONLY a JSON array, one object per pair:
[
  {"pair_index": 0, "at": "TRUE|PROBABLE|FALSE", "isAt": "TRUE|FALSE", "why": "≤15 words"},
  ...
]
Include every pair_index exactly once. No prose outside the array."""


# ============================================================
# Wikidata helpers (same as v6/v7)
# ============================================================

def load_wikidata_cache(path="wikidata_cache.json"):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"persons": {}, "locations": {}}


def get_person_desc(qid, wikidata_cache):
    pi = wikidata_cache.get("persons", {}).get(qid)
    if not pi: return ""
    parts = []
    if pi.get("occupations"): parts.append(pi["occupations"][0])
    if pi.get("nationalities"): parts.append(pi["nationalities"][0])
    if pi.get("birth_date"): parts.append(f"born {pi['birth_date'][:4]}")
    if pi.get("death_date"): parts.append(f"died {pi['death_date'][:4]}")
    if pi.get("birth_place"): parts.append(f"born in {pi['birth_place']}")
    return f" [{', '.join(parts)}]" if parts else ""


def get_location_desc(qid, wikidata_cache):
    li = wikidata_cache.get("locations", {}).get(qid)
    if not li: return ""
    parts = []
    if li.get("instance_types"): parts.append(li["instance_types"][0])
    if li.get("country"): parts.append(f"in {li['country']}")
    return f" [{', '.join(parts)}]" if parts else ""


# ============================================================
# Prompt builder
# ============================================================

def build_user_prompt(doc, wikidata_cache=None, use_wikidata=True):
    pairs_text = []
    for i, pair in enumerate(doc['sampled_pairs']):
        pers = json.dumps(pair['pers_mentions_list'], ensure_ascii=False)
        loc = json.dumps(pair['loc_mentions_list'], ensure_ascii=False)

        pers_desc = ""
        loc_desc = ""
        if use_wikidata and wikidata_cache:
            pq = pair.get('pers_wikidata_QID', '')
            lq = pair.get('loc_wikidata_QID', '')
            pers_desc = get_person_desc(pq, wikidata_cache)
            loc_desc = get_location_desc(lq, wikidata_cache)

        pairs_text.append(f"[{i}] Person: {pers}{pers_desc} — Place: {loc}{loc_desc}")

    return f"""Document date: {doc['date']}
Language: {doc.get('language', 'en')}

Document text:
\"\"\"
{doc['text']}
\"\"\"

Person-place pairs to classify:
{chr(10).join(pairs_text)}

Classify each pair following the definitions and examples above. Return ONLY the JSON array."""


# ============================================================
# API caller — supports Opus 4.7 (no temperature)
# ============================================================

def call_anthropic_api(system_prompt, user_prompt, api_key,
                        model="claude-sonnet-4-20250514",
                        max_retries=3, max_tokens=8192):
    import urllib.request, urllib.error

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    # Opus 4.7+ does NOT support temperature/top_p/top_k
    if not model.startswith("claude-opus-4-7"):
        payload["temperature"] = 0.0

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages",
                data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["content"][0]["text"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.warning(f"HTTP {e.code} attempt {attempt+1}: {body[:300]}")
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** (attempt + 1) * 5)
            else:
                raise
        except Exception as e:
            logger.warning(f"Error attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def parse_response(text, num_pairs):
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                results = json.loads(m.group())
            except:
                return [{"at": "FALSE", "isAt": "FALSE"} for _ in range(num_pairs)]
        else:
            return [{"at": "FALSE", "isAt": "FALSE"} for _ in range(num_pairs)]
    preds = [{"at": "FALSE", "isAt": "FALSE"} for _ in range(num_pairs)]
    for item in results:
        idx = item.get("pair_index", -1)
        if 0 <= idx < num_pairs:
            at = str(item.get("at", "FALSE")).upper()
            isAt = str(item.get("isAt", "FALSE")).upper()
            if at not in ("TRUE", "PROBABLE", "FALSE"): at = "FALSE"
            if isAt not in ("TRUE", "FALSE"): isAt = "FALSE"
            if at == "FALSE": isAt = "FALSE"
            preds[idx] = {
                "at": at, "isAt": isAt,
                "at_explanation": item.get("why", ""),
                "isAt_explanation": item.get("why", ""),
            }
    return preds


def process_document(doc, api_key, model="claude-sonnet-4-20250514",
                      wikidata_cache=None, use_wikidata=True):
    user_prompt = build_user_prompt(doc, wikidata_cache, use_wikidata)
    response = call_anthropic_api(SYSTEM_PROMPT, user_prompt, api_key, model)
    preds = parse_response(response, len(doc['sampled_pairs']))

    out_doc = {k: doc[k] for k in ["document_id", "media", "source", "date", "language", "text"] if k in doc}
    out_doc["sampled_pairs"] = []
    for i, pair in enumerate(doc['sampled_pairs']):
        p = preds[i]
        out_doc["sampled_pairs"].append({
            "pers_entity_id": pair["pers_entity_id"],
            "pers_wikidata_QID": pair.get("pers_wikidata_QID"),
            "pers_mentions_list": pair["pers_mentions_list"],
            "loc_entity_id": pair["loc_entity_id"],
            "loc_wikidata_QID": pair.get("loc_wikidata_QID"),
            "loc_mentions_list": pair["loc_mentions_list"],
            "at": p["at"], "isAt": p["isAt"],
            "at_explanation": p.get("at_explanation", ""),
            "isAt_explanation": p.get("isAt_explanation", ""),
        })
    return out_doc


# ============================================================
# Eval helpers — same as v6/v7
# ============================================================

def compute_macro_recall(gold, pred, labels):
    rec = {}
    for l in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == l and p == l)
        t = sum(1 for g in gold if g == l)
        rec[l] = tp / t if t else 0.0
    return sum(rec.values()) / len(labels), rec


def compute_confusion(gold, pred, labels):
    """Confusion matrix as nested dict: cm[gold_label][pred_label] = count"""
    cm = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        if g in cm and p in cm[g]:
            cm[g][p] += 1
    return cm


def evaluate(gold_docs, pred_docs):
    gold_at, pred_at = [], []
    gold_isAt, pred_isAt = [], []
    pmap = {d["document_id"]: d for d in pred_docs}
    for g in gold_docs:
        p = pmap.get(g["document_id"])
        if not p: continue
        for i, gp in enumerate(g["sampled_pairs"]):
            if i < len(p["sampled_pairs"]):
                pp = p["sampled_pairs"][i]
                gold_at.append(gp["at"]); pred_at.append(pp["at"])
                gold_isAt.append(gp["isAt"]); pred_isAt.append(pp["isAt"])
    at_m, at_r = compute_macro_recall(gold_at, pred_at, ["TRUE", "PROBABLE", "FALSE"])
    is_m, is_r = compute_macro_recall(gold_isAt, pred_isAt, ["TRUE", "FALSE"])
    cm_at = compute_confusion(gold_at, pred_at, ["TRUE", "PROBABLE", "FALSE"])
    cm_is = compute_confusion(gold_isAt, pred_isAt, ["TRUE", "FALSE"])
    return {
        "at": {"macro_recall": at_m, "per_label": at_r, "confusion": cm_at},
        "isAt": {"macro_recall": is_m, "per_label": is_r, "confusion": cm_is},
        "global_score": (at_m + is_m) / 2,
        "total_pairs": len(gold_at),
    }


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(docs, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for d in docs: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(docs)} docs to {path}")


def print_results(lang, m):
    print(f"\n{'='*60}")
    print(f"RESULTS — {lang.upper()}")
    print(f"{'='*60}")
    print(f"  Total pairs: {m['total_pairs']}")
    print(f"  at  macro recall:  {m['at']['macro_recall']:.4f}")
    for l, v in m['at']['per_label'].items():
        print(f"    {l:9s} {v:.4f}")
    print(f"  isAt macro recall: {m['isAt']['macro_recall']:.4f}")
    for l, v in m['isAt']['per_label'].items():
        print(f"    {l:9s} {v:.4f}")
    print(f"  GLOBAL SCORE:      {m['global_score']:.4f}")
    print(f"\n  at confusion matrix:")
    for g, row in m['at']['confusion'].items():
        s = "  ".join(f"{p}={c:3d}" for p, c in row.items())
        total = sum(row.values())
        print(f"    gold={g:9s} → pred: {s} (total={total})")
    print(f"  isAt confusion matrix:")
    for g, row in m['isAt']['confusion'].items():
        s = "  ".join(f"{p}={c:3d}" for p, c in row.items())
        total = sum(row.values())
        print(f"    gold={g:9s} → pred: {s} (total={total})")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "eval"], required=True)
    parser.add_argument("--input_file"); parser.add_argument("--data_dir")
    parser.add_argument("--output_dir", default="./output_v8")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max_docs", type=int)
    parser.add_argument("--lang")
    parser.add_argument("--wikidata", default="wikidata_cache.json")
    parser.add_argument("--no_wikidata", action="store_true")
    args = parser.parse_args()

    wikidata_cache = load_wikidata_cache(args.wikidata) if not args.no_wikidata else {}
    if wikidata_cache.get("persons"):
        logger.info(f"Wikidata: {len(wikidata_cache['persons'])} persons, {len(wikidata_cache['locations'])} locations")
    use_wikidata = not args.no_wikidata
    logger.info(f"Model: {args.model}  use_wikidata={use_wikidata}")

    if args.mode == "predict":
        files = [args.input_file] if args.input_file else sorted(Path(args.data_dir).glob("*.jsonl"))
        for fpath in files:
            docs = load_jsonl(str(fpath))
            if args.lang: docs = [d for d in docs if d['language'] == args.lang]
            if args.max_docs: docs = docs[:args.max_docs]

            out = []
            for i, d in enumerate(docs):
                logger.info(f"[{i+1}/{len(docs)}] {d['document_id']} ({len(d['sampled_pairs'])} pairs)")
                try:
                    out.append(process_document(d, args.api_key, args.model, wikidata_cache, use_wikidata))
                except Exception as e:
                    logger.error(f"Failed {d['document_id']}: {e}")
                    out.append(d)
            save_jsonl(out, os.path.join(args.output_dir, os.path.basename(str(fpath))))

    elif args.mode == "eval":
        langs = [args.lang] if args.lang else ["fr", "en", "de"]
        for lang in langs:
            # Try multiple naming conventions:
            # de-dev.jsonl, de-train.jsonl, *-de.jsonl, *train-de.jsonl
            data_dir = Path(args.data_dir)
            patterns = [f"{lang}-*.jsonl", f"*-{lang}.jsonl", f"*-{lang}-*.jsonl", f"*{lang}*.jsonl"]
            fp = []
            for pat in patterns:
                fp = list(data_dir.glob(pat))
                if fp: break
            if not fp:
                logger.warning(f"No file for {lang} in {data_dir}")
                continue

            if len(fp) > 1:
                dev_files = [f for f in fp if 'dev' in f.name.lower()]
                train_files = [f for f in fp if 'train' in f.name.lower() and 'dev' not in f.name.lower()]
                if dev_files:
                    fp = [dev_files[0]]
                    logger.info(f"Multiple {lang} files; using DEV: {fp[0].name}")
                elif train_files:
                    fp = [train_files[0]]
                    logger.info(f"Multiple {lang} files; using TRAIN: {fp[0].name}")
                else:
                    fp = [fp[0]]

            docs = load_jsonl(str(fp[0]))
            if args.max_docs: docs = docs[:args.max_docs]
            logger.info(f"\n{'='*60}\n{lang.upper()}: {len(docs)} docs, {sum(len(d['sampled_pairs']) for d in docs)} pairs\n{'='*60}")

            preds = []
            for i, d in enumerate(docs):
                logger.info(f"[{i+1}/{len(docs)}] {d['document_id']}")
                try:
                    preds.append(process_document(d, args.api_key, args.model, wikidata_cache, use_wikidata))
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    fb = {**d}
                    for p in fb['sampled_pairs']:
                        p['at'] = 'FALSE'
                        p['isAt'] = 'FALSE'
                    preds.append(fb)

            m = evaluate(docs, preds)
            print_results(lang, m)
            save_jsonl(preds, os.path.join(args.output_dir, f"eval-{lang}.jsonl"))


if __name__ == "__main__":
    main()