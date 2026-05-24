#!/usr/bin/env python3
"""
HIPE-2026 KG Reasoner
=====================
For each (person QID, location QID) pair in HIPE data, query Wikidata for
direct relations between them and apply deterministic rules.

Wikidata properties we care about:
  P19  place of birth      -> at=TRUE; isAt=TRUE if article_date ~ birth_date (rare)
  P20  place of death      -> at=TRUE; isAt=TRUE if article_date in [death-30d, death+30d]
  P551 residence           -> at=TRUE; isAt depends on time interval P580/P582
  P937 work location       -> at=TRUE; isAt depends on interval
  P39  position held at    -> at=PROBABLE (role does NOT necessarily mean physical presence)
  P27  citizenship country -> if location is country -> at=PROBABLE
  P569 date of birth       -> for sanity / temporal reasoning
  P570 date of death       -> if death < article_date -> isAt=FALSE GUARANTEED
  P31  instance of (location)
  P17  country (location)

The reasoner produces, for each pair, ONE of:
  - "deterministic": a label decided purely by KG facts (no LLM needed)
  - "advisory":      hints + facts that the LLM should consider
  - "no_signal":     no KG facts found, LLM decides alone

Output: kg_facts.jsonl  (one row per unique (pers_qid, loc_qid) pair across all docs)
        per_pair_decisions.jsonl  (one row per (doc_id, pair_idx) with the decision applied)

Usage:
  # First time: build full KG cache from sandbox data (one-shot)
  python kg_reasoner.py --build_cache \
      --inputs sandbox_dev/*.jsonl raw/*.jsonl \
      --cache kg_facts.jsonl

  # Then: apply to a specific data file (uses cache)
  python kg_reasoner.py --apply \
      --input sandbox_dev/HIPE-2026-sandbox-train-de.jsonl \
      --cache kg_facts.jsonl \
      --output kg_decisions_de.jsonl \
      --report kg_report_de.txt

Notes:
  - Wikidata SPARQL endpoint: https://query.wikidata.org/sparql
  - Rate limit: ~5 req/sec sustained, much higher for short queries.
  - We retry on 429, sleep 1s on success, 5s on 429, 30s on 5xx.
  - Cache is keyed by (pers_qid, loc_qid). Disk-persisted, resumable.
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
# Wikidata strongly recommends a descriptive User-Agent with contact info.
# Otherwise it returns 403/429 aggressively. Edit this if you fork the script.
USER_AGENT = (
    "HIPE2026-KG-Reasoner/1.0 "
    "(https://hipe-eval.github.io/HIPE-2026/; CLEF research participant) "
    "Python-urllib/3"
)

# ============================================================
# SPARQL queries
# ============================================================

# Simple direct relation queries — split into separate calls for robustness.
# A single big UNION query was timing out on some Wikidata endpoints.

PAIR_DIRECT_QUERY = """
SELECT ?relation WHERE {{
  {{
    wd:{pq} wdt:P19 wd:{lq} .
    BIND("birth_place" AS ?relation)
  }} UNION {{
    wd:{pq} wdt:P20 wd:{lq} .
    BIND("death_place" AS ?relation)
  }}
}}
"""

PAIR_RESIDENCE_QUERY = """
SELECT ?startDate ?endDate WHERE {{
  wd:{pq} p:P551 ?stmt .
  ?stmt ps:P551 wd:{lq} .
  OPTIONAL {{ ?stmt pq:P580 ?startDate . }}
  OPTIONAL {{ ?stmt pq:P582 ?endDate . }}
}}
LIMIT 5
"""

PAIR_WORK_QUERY = """
SELECT ?startDate ?endDate WHERE {{
  wd:{pq} p:P937 ?stmt .
  ?stmt ps:P937 wd:{lq} .
  OPTIONAL {{ ?stmt pq:P580 ?startDate . }}
  OPTIONAL {{ ?stmt pq:P582 ?endDate . }}
}}
LIMIT 5
"""

PAIR_POSITION_QUERY = """
SELECT ?startDate ?endDate WHERE {{
  wd:{pq} p:P39 ?stmt .
  ?stmt ps:P39 ?position .
  {{ ?position wdt:P937 wd:{lq} . }} UNION {{ ?position wdt:P276 wd:{lq} . }}
  OPTIONAL {{ ?stmt pq:P580 ?startDate . }}
  OPTIONAL {{ ?stmt pq:P582 ?endDate . }}
}}
LIMIT 5
"""

PAIR_CITIZENSHIP_QUERY = """
ASK {{
  wd:{pq} wdt:P27 ?country .
  wd:{lq} wdt:P17 ?country .
}}
"""

PAIR_BIRTH_COUNTRY_QUERY = """
ASK {{
  wd:{pq} wdt:P19 ?bp .
  ?bp wdt:P17 ?country .
  wd:{lq} wdt:P17 ?country .
  FILTER(?bp != wd:{lq})
}}
"""

# Person info query (kept; it works on its own)
PERSON_DATES_QUERY_TEMPLATE = """
SELECT ?birth ?death ?occupation WHERE {{
  VALUES ?P {{ wd:{pers_qid} }}
  OPTIONAL {{ ?P wdt:P569 ?birth . }}
  OPTIONAL {{ ?P wdt:P570 ?death . }}
  OPTIONAL {{ ?P wdt:P106 ?occ . ?occ rdfs:label ?occupation . FILTER(LANG(?occupation) = "en") }}
}}
LIMIT 5
"""

# Per location: type and country (cached per location)
LOCATION_INFO_QUERY_TEMPLATE = """
SELECT ?type ?country ?countryLabel WHERE {{
  VALUES ?L {{ wd:{loc_qid} }}
  OPTIONAL {{ ?L wdt:P31 ?type . }}
  OPTIONAL {{
    ?L wdt:P17 ?country .
    ?country rdfs:label ?countryLabel .
    FILTER(LANG(?countryLabel) = "en")
  }}
}}
LIMIT 5
"""


# ============================================================
# SPARQL caller with rate limiting + backoff
# ============================================================

def run_sparql(query: str, max_retries: int = 5) -> dict:
    """Run a SPARQL query (POST) and return the JSON results dict."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    body = urllib.parse.urlencode({"query": query, "format": "json"}).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(WIKIDATA_ENDPOINT, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                log.warning(f"429 rate-limited, sleeping {wait}s")
                time.sleep(wait)
            elif e.code in (403,):
                # Possibly UA issue. Show body once.
                try:
                    body_txt = e.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    body_txt = ""
                log.warning(f"403 Forbidden (try editing USER_AGENT). Body: {body_txt}")
                return {"results": {"bindings": []}}
            elif e.code >= 500:
                wait = 30
                log.warning(f"{e.code} server error, sleeping {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"HTTP {e.code}: {e.reason}")
                return {"results": {"bindings": []}}
        except (urllib.error.URLError, TimeoutError) as e:
            log.warning(f"Network error attempt {attempt+1}: {e}")
            time.sleep(5)

    log.error("Max SPARQL retries exceeded; returning empty.")
    return {"results": {"bindings": []}}


# ============================================================
# Disk-persisted cache
# ============================================================

class JsonlCache:
    """Append-only JSONL cache keyed by string. Resumable."""

    def __init__(self, path: str):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.data[row["_key"]] = row

    def has(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value: dict):
        value["_key"] = key
        self.data[key] = value
        with open(self.path, "a") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self.data)


# ============================================================
# Wikidata fetchers
# ============================================================

def _run_ask(query: str) -> bool:
    """Run an ASK query and return the boolean result."""
    res = run_sparql(query)
    return bool(res.get("boolean", False))


def fetch_pair_relations(pers_qid: str, loc_qid: str) -> dict:
    """Fetch all direct relations between a (person, location) pair.

    Uses 6 small queries instead of one big UNION (more robust on Wikidata).
    """
    if not pers_qid or not loc_qid or not pers_qid.startswith("Q") or not loc_qid.startswith("Q"):
        return {"relations": [], "error": "invalid_qid"}

    relations = []

    # 1) birth_place / death_place (single query, cheap)
    res = run_sparql(PAIR_DIRECT_QUERY.format(pq=pers_qid, lq=loc_qid))
    for b in res.get("results", {}).get("bindings", []):
        rel_type = b.get("relation", {}).get("value", "")
        if rel_type:
            relations.append({"type": rel_type})

    # 2) residence with date intervals
    res = run_sparql(PAIR_RESIDENCE_QUERY.format(pq=pers_qid, lq=loc_qid))
    for b in res.get("results", {}).get("bindings", []):
        rel = {"type": "residence"}
        if "startDate" in b: rel["start_date"] = b["startDate"]["value"]
        if "endDate" in b:   rel["end_date"]   = b["endDate"]["value"]
        relations.append(rel)

    # 3) work_location with date intervals
    res = run_sparql(PAIR_WORK_QUERY.format(pq=pers_qid, lq=loc_qid))
    for b in res.get("results", {}).get("bindings", []):
        rel = {"type": "work_location"}
        if "startDate" in b: rel["start_date"] = b["startDate"]["value"]
        if "endDate" in b:   rel["end_date"]   = b["endDate"]["value"]
        relations.append(rel)

    # 4) position_at with date intervals
    res = run_sparql(PAIR_POSITION_QUERY.format(pq=pers_qid, lq=loc_qid))
    for b in res.get("results", {}).get("bindings", []):
        rel = {"type": "position_at"}
        if "startDate" in b: rel["start_date"] = b["startDate"]["value"]
        if "endDate" in b:   rel["end_date"]   = b["endDate"]["value"]
        relations.append(rel)

    # 5/6) Soft signals (ASK queries — only run if no direct match yet, to save time)
    has_direct = any(r["type"] in ("birth_place", "death_place", "residence",
                                    "work_location", "position_at") for r in relations)
    if not has_direct:
        if _run_ask(PAIR_CITIZENSHIP_QUERY.format(pq=pers_qid, lq=loc_qid)):
            relations.append({"type": "citizenship_match"})
        if _run_ask(PAIR_BIRTH_COUNTRY_QUERY.format(pq=pers_qid, lq=loc_qid)):
            relations.append({"type": "birth_country_match"})

    return {"relations": relations}


def fetch_person_info(pers_qid: str) -> dict:
    """Fetch person metadata: dates and occupations."""
    if not pers_qid or not pers_qid.startswith("Q"):
        return {}
    query = PERSON_DATES_QUERY_TEMPLATE.format(pers_qid=pers_qid)
    res = run_sparql(query)
    info = {"birth_date": None, "death_date": None, "occupations": []}
    for b in res.get("results", {}).get("bindings", []):
        if "birth" in b and not info["birth_date"]:
            info["birth_date"] = b["birth"]["value"]
        if "death" in b and not info["death_date"]:
            info["death_date"] = b["death"]["value"]
        if "occupation" in b:
            occ = b["occupation"]["value"]
            if occ not in info["occupations"]:
                info["occupations"].append(occ)
    return info


def fetch_location_info(loc_qid: str) -> dict:
    """Fetch location metadata: type and country."""
    if not loc_qid or not loc_qid.startswith("Q"):
        return {}
    query = LOCATION_INFO_QUERY_TEMPLATE.format(loc_qid=loc_qid)
    res = run_sparql(query)
    info = {"types": [], "country": None}
    for b in res.get("results", {}).get("bindings", []):
        if "type" in b:
            t = b["type"]["value"].split("/")[-1]
            if t not in info["types"]:
                info["types"].append(t)
        if "countryLabel" in b and not info["country"]:
            info["country"] = b["countryLabel"]["value"]
    return info


# ============================================================
# Date parsing & comparison
# ============================================================

def parse_wd_date(s: str):
    """Parse Wikidata-style date '1769-08-15T00:00:00Z' or '+1769-08-15T...' to (year, month, day) tuple."""
    if not s:
        return None
    s = s.lstrip("+")
    try:
        # ISO format
        date_part = s.split("T")[0]
        y, m, d = date_part.split("-")
        return (int(y), int(m), int(d))
    except (ValueError, IndexError):
        return None


def parse_article_date(s: str):
    """HIPE article date is e.g. '1885-03-15'."""
    if not s:
        return None
    try:
        y, m, d = s.split("-")
        return (int(y), int(m), int(d))
    except (ValueError, IndexError):
        # Maybe just YYYY
        try:
            return (int(s), 1, 1)
        except ValueError:
            return None


def date_lt(a, b):
    """a < b (strictly before)."""
    return a is not None and b is not None and a < b


def date_in_range(d, start, end):
    """d falls in [start, end], with None meaning unbounded."""
    if d is None:
        return False
    if start is not None and d < start:
        return False
    if end is not None and d > end:
        return False
    return True


# ============================================================
# Decision logic
# ============================================================

# Hierarchy: stronger relation wins for `at`
RELATION_AT_PRIORITY = {
    "birth_place":         ("TRUE", 100, "P19_birth"),
    "death_place":         ("TRUE", 95,  "P20_death"),
    "residence":           ("TRUE", 80,  "P551_residence"),
    "work_location":       ("TRUE", 75,  "P937_work"),
    "position_at":         ("PROBABLE", 60, "P39_position"),
    "citizenship_match":   ("PROBABLE", 30, "P27_citizenship"),
    "birth_country_match": ("PROBABLE", 25, "P19_country"),
}


def decide_from_kg(pair_relations: dict, person_info: dict,
                    article_date_str: str) -> dict:
    """
    Apply deterministic rules to decide (at, isAt) from KG facts.

    Returns dict with:
      decision_type: 'deterministic_strong' | 'deterministic_weak' | 'advisory' | 'no_signal'
      at: TRUE/PROBABLE/FALSE/None
      isAt: TRUE/FALSE/None
      rule: human-readable rule name
      facts: list of facts used
      hints: text hints for LLM
    """
    facts = pair_relations.get("relations", [])
    article_date = parse_article_date(article_date_str)
    death_date = parse_wd_date(person_info.get("death_date"))
    birth_date = parse_wd_date(person_info.get("birth_date"))

    # ---- Hard temporal constraint: dead person → isAt = FALSE always
    person_dead = death_date is not None and article_date is not None and date_lt(death_date, article_date)
    person_not_yet_born = birth_date is not None and article_date is not None and date_lt(article_date, birth_date)

    if not facts:
        # No direct (P, L) relation in KG
        if person_not_yet_born:
            # Person hadn't been born yet → at = FALSE absolutely
            return {
                "decision_type": "deterministic_strong",
                "at": "FALSE", "isAt": "FALSE",
                "rule": "person_not_born_yet",
                "facts": [], "hints": ["person not yet born at article date"],
            }
        return {
            "decision_type": "no_signal",
            "at": None, "isAt": None,
            "rule": None,
            "facts": [], "hints": [],
        }

    # ---- Pick strongest fact for `at`
    best = None  # (priority, label, rule, fact)
    for f in facts:
        info = RELATION_AT_PRIORITY.get(f["type"])
        if not info:
            continue
        label, prio, rule = info
        if best is None or prio > best[0]:
            best = (prio, label, rule, f)

    if not best:
        return {
            "decision_type": "no_signal",
            "at": None, "isAt": None,
            "rule": None,
            "facts": facts, "hints": [],
        }

    _, at_label, at_rule, best_fact = best
    rel_type = best_fact["type"]

    # ---- Decide isAt
    isat_label = "FALSE"  # default
    isat_rule = "default_false"

    if person_dead:
        isat_label = "FALSE"
        isat_rule = "person_dead_at_pub"
    elif person_not_yet_born:
        # Should have been caught above; defensive
        return {
            "decision_type": "deterministic_strong",
            "at": "FALSE", "isAt": "FALSE",
            "rule": "person_not_born_yet",
            "facts": facts, "hints": ["person not yet born"],
        }
    else:
        # Check if relation has a time interval covering article date
        start = parse_wd_date(best_fact.get("start_date"))
        end = parse_wd_date(best_fact.get("end_date"))

        if rel_type == "death_place":
            # isAt=TRUE only if article published within ~30 days of death
            if death_date and article_date:
                # Within 60 days (approx, ignoring month variations)
                # We compute absolute year-month-day diff conservatively.
                if abs(article_date[0] - death_date[0]) == 0 and \
                   abs(article_date[1] - death_date[1]) <= 1:
                    isat_label = "TRUE"
                    isat_rule = "death_within_2_months"
        elif rel_type in ("residence", "work_location"):
            # isAt=TRUE if interval covers article_date
            if date_in_range(article_date, start, end):
                isat_label = "TRUE"
                isat_rule = f"{rel_type}_active_at_pub"
        elif rel_type == "position_at":
            # Position implies role, NOT necessarily presence at the place at the time
            # Keep at=PROBABLE, isAt=FALSE unless interval covers
            if date_in_range(article_date, start, end):
                # Even so, position alone is weak for current presence
                isat_label = "FALSE"
                isat_rule = "position_active_but_not_presence"
        elif rel_type == "birth_place":
            # Born in L. Not necessarily at L now.
            # Adult? Likely not present.
            if birth_date and article_date:
                age_years = article_date[0] - birth_date[0]
                if age_years < 5:
                    isat_label = "TRUE"
                    isat_rule = "birth_place_recent"
                else:
                    isat_label = "FALSE"
                    isat_rule = "birth_place_long_ago"

    decision_type = "deterministic_strong" if at_rule in ("P19_birth", "P20_death") \
                    else "deterministic_weak"

    hints = [f"KG: {f['type']}" for f in facts]
    if person_dead:
        hints.append(f"person died {death_date} before article {article_date}")

    return {
        "decision_type": decision_type,
        "at": at_label,
        "isAt": isat_label,
        "rule": f"at:{at_rule}|isAt:{isat_rule}",
        "facts": facts,
        "hints": hints,
        "person_dead": person_dead,
    }


# ============================================================
# Main: build cache & apply
# ============================================================

def collect_qid_pairs(input_files):
    """From HIPE jsonl files, collect all unique (pers_qid, loc_qid) pairs and unique persons/locations."""
    pairs = set()
    persons = set()
    locations = set()
    pair_to_docs = defaultdict(list)  # for stats

    for path in input_files:
        log.info(f"Scanning {path}...")
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                doc_id = d.get("document_id")
                for i, p in enumerate(d.get("sampled_pairs", [])):
                    pq = p.get("pers_wikidata_QID") or ""
                    lq = p.get("loc_wikidata_QID") or ""
                    if pq.startswith("Q"):
                        persons.add(pq)
                    if lq.startswith("Q"):
                        locations.add(lq)
                    if pq.startswith("Q") and lq.startswith("Q"):
                        pairs.add((pq, lq))
                        pair_to_docs[(pq, lq)].append((doc_id, i))
    return pairs, persons, locations, pair_to_docs


def build_cache(input_files, cache_path: str, sleep_between=0.2):
    """Fetch all KG data we need and persist to cache."""
    pair_cache = JsonlCache(cache_path)
    person_cache = JsonlCache(cache_path.replace(".jsonl", "_persons.jsonl"))
    location_cache = JsonlCache(cache_path.replace(".jsonl", "_locations.jsonl"))

    pairs, persons, locations, _ = collect_qid_pairs(input_files)
    log.info(f"Unique pairs: {len(pairs)}, persons: {len(persons)}, locations: {len(locations)}")
    log.info(f"Already cached: {len(pair_cache)} pairs, {len(person_cache)} persons, {len(location_cache)} locations")

    # 1) Person info
    todo_persons = [p for p in persons if not person_cache.has(p)]
    log.info(f"Fetching {len(todo_persons)} persons...")
    for i, pq in enumerate(todo_persons):
        info = fetch_person_info(pq)
        person_cache.put(pq, info)
        if (i + 1) % 50 == 0:
            log.info(f"  persons {i+1}/{len(todo_persons)}")
        time.sleep(sleep_between)

    # 2) Location info
    todo_locations = [l for l in locations if not location_cache.has(l)]
    log.info(f"Fetching {len(todo_locations)} locations...")
    for i, lq in enumerate(todo_locations):
        info = fetch_location_info(lq)
        location_cache.put(lq, info)
        if (i + 1) % 50 == 0:
            log.info(f"  locations {i+1}/{len(todo_locations)}")
        time.sleep(sleep_between)

    # 3) Pair relations
    todo_pairs = [p for p in pairs if not pair_cache.has(f"{p[0]}|{p[1]}")]
    log.info(f"Fetching {len(todo_pairs)} pair relations...")
    for i, (pq, lq) in enumerate(todo_pairs):
        rels = fetch_pair_relations(pq, lq)
        pair_cache.put(f"{pq}|{lq}", rels)
        if (i + 1) % 100 == 0:
            log.info(f"  pairs {i+1}/{len(todo_pairs)}")
        time.sleep(sleep_between)

    log.info(f"DONE. Cache contains {len(pair_cache)} pairs, {len(person_cache)} persons, {len(location_cache)} locations.")
    return pair_cache, person_cache, location_cache


def apply_decisions(input_file: str, cache_path: str, output_path: str, report_path: str = None):
    """Apply KG decisions to a HIPE input file, output decisions per (doc_id, pair_idx)."""
    pair_cache = JsonlCache(cache_path)
    person_cache = JsonlCache(cache_path.replace(".jsonl", "_persons.jsonl"))

    if len(pair_cache) == 0:
        log.error(f"Pair cache is empty: {cache_path}. Run --build_cache first.")
        sys.exit(1)

    decisions = []
    type_counter = Counter()
    rule_counter = Counter()
    coverage_counter = Counter()
    label_dist_at = Counter()
    label_dist_isat = Counter()

    with open(input_file) as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            doc_id = doc.get("document_id")
            article_date = doc.get("date", "")
            for i, p in enumerate(doc.get("sampled_pairs", [])):
                pq = p.get("pers_wikidata_QID") or ""
                lq = p.get("loc_wikidata_QID") or ""

                rels = pair_cache.get(f"{pq}|{lq}") if pq and lq else None
                pinfo = person_cache.get(pq) if pq else None

                if rels is None:
                    rels = {"relations": []}
                if pinfo is None:
                    pinfo = {}

                dec = decide_from_kg(rels, pinfo, article_date)
                row = {
                    "document_id": doc_id,
                    "pair_idx": i,
                    "pers_qid": pq,
                    "loc_qid": lq,
                    "article_date": article_date,
                    "decision_type": dec["decision_type"],
                    "at": dec["at"],
                    "isAt": dec["isAt"],
                    "rule": dec["rule"],
                    "facts": dec["facts"],
                    "hints": dec["hints"],
                    "person_dead": dec.get("person_dead", False),
                    # Gold labels (kept for evaluation)
                    "gold_at": p.get("at"),
                    "gold_isAt": p.get("isAt"),
                }
                decisions.append(row)
                type_counter[dec["decision_type"]] += 1
                if dec["rule"]:
                    rule_counter[dec["rule"]] += 1
                if dec["at"]:
                    label_dist_at[dec["at"]] += 1
                if dec["isAt"]:
                    label_dist_isat[dec["isAt"]] += 1

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in decisions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(decisions)} decisions to {output_path}")

    # Build report (coverage + accuracy on perechi where we have a deterministic decision)
    total = len(decisions)
    deterministic = sum(1 for d in decisions if d["decision_type"].startswith("deterministic"))
    no_signal = type_counter["no_signal"]

    # Accuracy of deterministic decisions vs gold
    det_at_correct = 0
    det_at_total = 0
    det_isat_correct = 0
    det_isat_total = 0
    for d in decisions:
        if d["decision_type"].startswith("deterministic"):
            if d["at"] is not None and d["gold_at"] is not None:
                det_at_total += 1
                if d["at"] == d["gold_at"]:
                    det_at_correct += 1
            if d["isAt"] is not None and d["gold_isAt"] is not None:
                det_isat_total += 1
                if d["isAt"] == d["gold_isAt"]:
                    det_isat_correct += 1

    report = []
    report.append(f"=== KG Reasoner Report: {input_file} ===")
    report.append(f"Total pairs: {total}")
    report.append(f"")
    report.append(f"Decision-type distribution:")
    for t, c in type_counter.most_common():
        report.append(f"  {t:25s} {c:5d}  ({100*c/total:5.1f}%)")
    report.append(f"")
    report.append(f"KG coverage (any deterministic decision): {deterministic}/{total} = {100*deterministic/total:.1f}%")
    report.append(f"")
    if det_at_total:
        report.append(f"Deterministic 'at'   accuracy: {det_at_correct}/{det_at_total} = {100*det_at_correct/det_at_total:.1f}%")
    if det_isat_total:
        report.append(f"Deterministic 'isAt' accuracy: {det_isat_correct}/{det_isat_total} = {100*det_isat_correct/det_isat_total:.1f}%")
    report.append(f"")
    report.append(f"Top rules used:")
    for r, c in rule_counter.most_common(15):
        report.append(f"  {r:50s} {c:5d}")
    report.append(f"")
    report.append(f"Predicted 'at' label distribution (deterministic only):")
    for l, c in label_dist_at.most_common():
        report.append(f"  {l:10s} {c:5d}")
    report.append(f"Predicted 'isAt' label distribution (deterministic only):")
    for l, c in label_dist_isat.most_common():
        report.append(f"  {l:10s} {c:5d}")

    # Per-rule accuracy breakdown (most useful for paper ablation table)
    rule_accuracy = defaultdict(lambda: {"at_correct": 0, "at_total": 0, "isat_correct": 0, "isat_total": 0})
    for d in decisions:
        if d["decision_type"].startswith("deterministic") and d["rule"]:
            r = d["rule"]
            if d["at"] is not None and d["gold_at"] is not None:
                rule_accuracy[r]["at_total"] += 1
                if d["at"] == d["gold_at"]:
                    rule_accuracy[r]["at_correct"] += 1
            if d["isAt"] is not None and d["gold_isAt"] is not None:
                rule_accuracy[r]["isat_total"] += 1
                if d["isAt"] == d["gold_isAt"]:
                    rule_accuracy[r]["isat_correct"] += 1

    report.append(f"")
    report.append(f"Per-rule accuracy (for paper ablation):")
    report.append(f"{'rule':50s} {'at_acc':>10s} {'isat_acc':>10s}")
    for r, s in sorted(rule_accuracy.items(), key=lambda x: -x[1]["at_total"]):
        at_acc = f"{100*s['at_correct']/s['at_total']:.1f}% ({s['at_correct']}/{s['at_total']})" if s['at_total'] else "n/a"
        isat_acc = f"{100*s['isat_correct']/s['isat_total']:.1f}% ({s['isat_correct']}/{s['isat_total']})" if s['isat_total'] else "n/a"
        report.append(f"  {r:50s} {at_acc:>20s} {isat_acc:>20s}")

    text = "\n".join(report)
    print(text)
    if report_path:
        with open(report_path, "w") as f:
            f.write(text)
        log.info(f"Wrote report to {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_cache", action="store_true", help="Fetch KG data into cache")
    ap.add_argument("--apply", action="store_true", help="Apply KG decisions to a HIPE file")
    ap.add_argument("--inputs", nargs="+", help="HIPE input files (for build_cache)")
    ap.add_argument("--input", help="HIPE input file (for apply)")
    ap.add_argument("--cache", default="kg_facts.jsonl")
    ap.add_argument("--output", help="Output decisions JSONL (for apply)")
    ap.add_argument("--report", help="Report text file (for apply)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between SPARQL calls")
    args = ap.parse_args()

    if args.build_cache:
        if not args.inputs:
            ap.error("--build_cache requires --inputs")
        # Expand globs
        files = []
        for pat in args.inputs:
            if "*" in pat:
                from glob import glob
                files.extend(glob(pat))
            else:
                files.append(pat)
        if not files:
            ap.error("No files matched --inputs")
        log.info(f"Will scan {len(files)} files: {files}")
        build_cache(files, args.cache, sleep_between=args.sleep)

    if args.apply:
        if not args.input or not args.output:
            ap.error("--apply requires --input and --output")
        apply_decisions(args.input, args.cache, args.output, args.report)


if __name__ == "__main__":
    main()