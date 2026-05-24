#!/usr/bin/env python3
"""
HIPE-2026 Dataset
=================
Unified Pytorch Dataset for the HIPE-2026 person-place relation task.

For each (document, pair) we build:
  - input_text: a focused window of text around the pair mentions, plus pair info
  - kg_features: a fixed-length numeric vector built from Wikidata facts
  - at_label: 0/1/2 for FALSE/PROBABLE/TRUE  (or -1 if unlabeled)
  - isAt_label: 0/1 for FALSE/TRUE  (or -1 if unlabeled)

The same class is used for train/dev/test by toggling `has_labels`.

Test data has no labels — we set them to -1, which the loss function ignores.
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Lazy torch import — Dataset class uses it, but feature-building utilities
# don't, so other tools (calibrator, ensemble) can import constants/helpers
# without requiring torch.
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    class _TorchDataset:
        pass


AT_LABELS = ["FALSE", "PROBABLE", "TRUE"]
ISAT_LABELS = ["FALSE", "TRUE"]
AT_LABEL2ID = {l: i for i, l in enumerate(AT_LABELS)}
ISAT_LABEL2ID = {l: i for i, l in enumerate(ISAT_LABELS)}

# Number of KG features (must match build_kg_features below)
KG_FEATURE_DIM = 16

# Number of FOL text-pattern features (must match build_text_pattern_features below)
# Order: [action_flag, role_flag, origin_flag, temporal_flag, departure_flag, candidacy_flag]
FOL_FEATURE_DIM = 6


# ============================================================
# Date utilities
# ============================================================

def parse_wd_date(s):
    """Parse '+1769-08-15T00:00:00Z' or '1769-08-15T00:00:00Z' → (y, m, d)."""
    if not s: return None
    s = s.lstrip("+")
    try:
        return tuple(int(x) for x in s.split("T")[0].split("-"))
    except (ValueError, IndexError):
        return None


def parse_article_date(s):
    if not s: return None
    try:
        y, m, d = s.split("-")
        return (int(y), int(m), int(d))
    except (ValueError, IndexError):
        try: return (int(s), 1, 1)
        except ValueError: return None


# ============================================================
# Text context extraction (OCR-tolerant)
# ============================================================

def normalize_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


def find_mention_positions(text_lower, mention):
    """Find char positions of mention in text. Tolerant to OCR whitespace differences."""
    mention = normalize_space(mention).lower()
    if not mention:
        return []
    positions = [(m.start(), m.end()) for m in re.finditer(re.escape(mention), text_lower)]
    if positions:
        return positions
    # Relaxed: whitespace -> \s+
    pattern = re.escape(mention)
    pattern = re.sub(r"\\\s+", r"\\s+", pattern)
    try:
        return [(m.start(), m.end()) for m in re.finditer(pattern, text_lower, flags=re.I)]
    except re.error:
        return []


def get_pair_context(text, pers_mentions, loc_mentions, window=400):
    """Return a focused window of text around the closest pers-loc mention pair."""
    text_lower = text.lower()
    p_spans = []
    for m in (pers_mentions or []):
        p_spans.extend(find_mention_positions(text_lower, m))
    l_spans = []
    for m in (loc_mentions or []):
        l_spans.extend(find_mention_positions(text_lower, m))

    if p_spans and l_spans:
        # Find closest pair of spans
        best = None
        for ps in p_spans:
            for ls in l_spans:
                # distance = gap between spans (0 if overlapping)
                if max(ps[0], ls[0]) <= min(ps[1], ls[1]):
                    dist = 0
                else:
                    dist = min(abs(ps[0] - ls[1]), abs(ls[0] - ps[1]))
                lo = min(ps[0], ls[0])
                hi = max(ps[1], ls[1])
                cand = (dist, lo, hi)
                if best is None or cand < best:
                    best = cand
        _, lo, hi = best
        left = max(0, lo - window)
        right = min(len(text), hi + window)
        return normalize_space(text[left:right])

    # Fallback: head + tail
    text = normalize_space(text)
    if len(text) <= 2 * window:
        return text
    return text[:window] + " [...] " + text[-window:]


def min_char_distance(text, pers_mentions, loc_mentions):
    """Min char distance between any pers and loc mention. 999999 if none found."""
    text_lower = text.lower()
    p_spans = []
    for m in (pers_mentions or []):
        p_spans.extend(find_mention_positions(text_lower, m))
    l_spans = []
    for m in (loc_mentions or []):
        l_spans.extend(find_mention_positions(text_lower, m))
    if not p_spans or not l_spans:
        return 999999
    best = 999999
    for ps in p_spans:
        for ls in l_spans:
            if max(ps[0], ls[0]) <= min(ps[1], ls[1]):
                return 0
            d = min(abs(ps[0] - ls[1]), abs(ls[0] - ps[1]))
            if d < best:
                best = d
    return best


def same_sentence_flag(text, pers_mentions, loc_mentions):
    """1 if any pers and loc mentions are in the same 'sentence' (no [.!?] between)."""
    text_lower = text.lower()
    p_spans = []
    for m in (pers_mentions or []):
        p_spans.extend(find_mention_positions(text_lower, m))
    l_spans = []
    for m in (loc_mentions or []):
        l_spans.extend(find_mention_positions(text_lower, m))
    if not p_spans or not l_spans:
        return 0
    for ps in p_spans:
        for ls in l_spans:
            lo = min(ps[0], ls[0])
            hi = max(ps[1], ls[1])
            span = text[lo:hi]
            if not re.search(r"[.!?]\s+[A-ZÄÖÜÉÈÀÊÎÔÛŒ]", span):
                return 1
    return 0


# ============================================================
# Regex syntactic markers (multilingual)
# ============================================================

# These are intentionally broad — calibrator/training will learn what matters.

ROLE_MARKERS = [
    r'\bdeput\w+', r'\bminister\w*', r'\bambassador\w*', r'\bsenator\w*', r'\bgouverneur\w*',
    r'\bgovernor\w*', r'\bpräsident\w*', r'\bpresident\w*', r'\bpremier\b',
    r'\bdéput\w+', r'\bministre\b', r'\bambassad\w+', r'\bmaire\b',
    r'\bAbgeordnet\w+', r'\bBürgermeister\w*', r'\bBotschafter\w*',
    r'\bgéneral\w*', r'\bGeneral\w*',
]

ORIGIN_MARKERS = [
    r'\bvon\b', r'\baus\b',                       # German
    r'\bof\b', r'\bfrom\b',                       # English
    r'\bde\b', r'\bdu\b', r'\bdes\b',             # French
]

ACTION_VERBS = [
    # presence/arrival
    r'\bist in\b', r'\bbefindet sich\b', r'\bweilte\b', r'\bankommen\w*\b',
    r'\bis in\b', r'\barrived\b', r'\bspoke\b', r'\bvisited\b', r'\bappeared\b',
    r'\bse trouve\b', r'\barrive\w*\b', r'\bs\'est rendu\b', r'\baux\b',
    # departure
    r'\bverließ\b', r'\bleft\b', r'\bdeparted\b', r'\ba quitté\b',
]

DATELINE_PATTERN = re.compile(
    r'\b(?:[A-Z][A-Z\-]{2,}|[A-ZÄÖÜ][a-zäöüß]+)\s*,\s*\d{1,2}\.?\s*(?:'
    r'Jan(?:uary|uar|vier)?|Feb(?:ruary|ruar|vrier)?|Mar(?:ch|s|z|ärz)?|'
    r'Apr(?:il)?|Mai|May|Jun[ie]?|Jul[iy]?|Aug(?:ust)?|Sep(?:tember|t)?|'
    r'Oct(?:ober|obre)?|Okt(?:ober)?|Nov(?:ember|embre)?|Dec(?:ember|embre)?|Dez(?:ember)?'
    r')\b'
)


def syntactic_features(context_text):
    """Return a dict of regex-based syntactic feature flags."""
    t = context_text
    return {
        "has_role": int(any(re.search(p, t, re.I) for p in ROLE_MARKERS)),
        "has_origin": int(any(re.search(p, t) for p in ORIGIN_MARKERS)),
        "has_action_verb": int(any(re.search(p, t, re.I) for p in ACTION_VERBS)),
        "has_dateline": int(bool(DATELINE_PATTERN.search(t))),
    }


# ============================================================
# KG feature extraction
# ============================================================

def build_kg_features(pers_qid, loc_qid, article_date_str,
                      kg_pairs=None, kg_persons=None, kg_locations=None):
    """
    Return a fixed-length numeric vector of KG features.

    Order (16 dims):
      [0]  has_kg_person       (1 if person QID has Wikidata info)
      [1]  has_kg_location     (1 if location QID has Wikidata info)
      [2]  person_dead_at_pub  (1 if death_date < article_date, else 0)
      [3]  person_unborn_at_pub (1 if birth_date > article_date, else 0)
      [4]  person_alive_at_pub (1 if alive)
      [5]  has_birth_place_rel (1 if pair has P19 to L)
      [6]  has_death_place_rel (1 if pair has P20 to L)
      [7]  has_residence_rel   (1 if pair has P551 to L)
      [8]  has_work_rel        (1 if pair has P937 to L)
      [9]  has_position_rel    (1 if pair has P39 with location L)
      [10] has_citizenship_rel (1 if person citizen of L's country)
      [11] num_relations       (count, normalized: count/5)
      [12] location_in_country (1 if L has a country property)
      [13] num_occupations     (count, normalized: count/3)
      [14] year_diff_birth_pub (article_year - birth_year, normalized: /100)
      [15] year_diff_death_pub (article_year - death_year, normalized: /100, 0 if unknown)
    """
    feats = [0.0] * 16

    if not kg_pairs: kg_pairs = {}
    if not kg_persons: kg_persons = {}
    if not kg_locations: kg_locations = {}

    pinfo = kg_persons.get(pers_qid, {})
    linfo = kg_locations.get(loc_qid, {})
    rels = kg_pairs.get(f"{pers_qid}|{loc_qid}", {}).get("relations", []) if pers_qid and loc_qid else []

    feats[0] = 1.0 if pinfo else 0.0
    feats[1] = 1.0 if linfo else 0.0

    # Temporal
    article_date = parse_article_date(article_date_str)
    death_date = parse_wd_date(pinfo.get("death_date"))
    birth_date = parse_wd_date(pinfo.get("birth_date"))

    if death_date and article_date:
        feats[2] = 1.0 if death_date < article_date else 0.0
    if birth_date and article_date:
        feats[3] = 1.0 if article_date < birth_date else 0.0
    if article_date and birth_date and death_date:
        feats[4] = 1.0 if (birth_date <= article_date <= death_date) else 0.0
    elif article_date and birth_date and not death_date:
        feats[4] = 1.0 if birth_date <= article_date else 0.0

    # Relation flags
    rel_types = [r["type"] for r in rels]
    feats[5] = 1.0 if "birth_place" in rel_types else 0.0
    feats[6] = 1.0 if "death_place" in rel_types else 0.0
    feats[7] = 1.0 if "residence" in rel_types else 0.0
    feats[8] = 1.0 if "work_location" in rel_types else 0.0
    feats[9] = 1.0 if "position_at" in rel_types else 0.0
    feats[10] = 1.0 if "citizenship_match" in rel_types else 0.0
    feats[11] = min(len(rels) / 5.0, 1.0)
    feats[12] = 1.0 if linfo.get("country") else 0.0
    feats[13] = min(len(pinfo.get("occupations", [])) / 3.0, 1.0)

    if article_date and birth_date:
        feats[14] = (article_date[0] - birth_date[0]) / 100.0
    if article_date and death_date:
        feats[15] = (article_date[0] - death_date[0]) / 100.0

    return feats


# ============================================================
# FOL text pattern features
# ============================================================
# These features detect linguistic patterns in the local context around
# the (person, place) mention. They are used in two ways:
#   1. As input features to the model (additional signal)
#   2. As constraint-loss inputs during training (penalize violations)

import re

# Patterns to detect, per language. We use simple word-boundary regex
# robust to OCR. Multilingual to handle DE/EN/FR/LU.

# Action verbs (locative actions) — strong signal for at=TRUE
_ACTION_PATTERNS = {
    'en': r'\b(arrived|arriving|spoke|speaks|speaking|residing|resided|departed|departing|left|leaves|received|receiving|attended|attending|passed through|stayed|staying|visited|visiting|fled|fleeing|marched|marching|settled|reached|toured|inaugurated|opened|closed|died|born|crowned|elected at|appointed to|presided)\b',
    'fr': r'\b(arriva|arrivé|parla|parlant|prononça|prononcé|résida|résidait|partit|partait|quitta|quitté|reçu|reçut|assista|séjourné|visita|visité|fuyit|marcha|s\'installa|s\'établit|atteignit|inaugura|inauguré|décéda|décédé|naquit|couronné|présida|prononce|fit|tint|donna)\b',
    'de': r'\b(ankam|ankommend|sprach|sprechen|residierte|wohnte|verließ|verlassen|empfing|empfangen|nahm teil|besuchte|floh|flüchtete|marschierte|niedergelassen|erreichte|eröffnete|starb|geboren|gekrönt|tagte|hielt|trat zurück|kam an)\b',
    'lb': r'\b(koum|koumen|sprach|wunnt|wunnen|asst do|empfangen|besicht|gestuerwen|gebuer)\b',
}

# Role/Position markers — strong signal for at=PROBABLE
_ROLE_PATTERNS = {
    'en': r'\b(ambassador|deputy|representative|minister|candidate|envoy|consul|governor|mayor|prefect|bishop|director|president|senator|congressman|councillor)\b',
    'fr': r'\b(ambassadeur|député|représentant|ministre|candidat|envoyé|consul|gouverneur|maire|préfet|évêque|directeur|président|sénateur|conseiller)\b',
    'de': r'\b(Botschafter|Abgeordneter|Abgeordneten|Vertreter|Minister|Kandidat|Konsul|Gouverneur|Bürgermeister|Präfekt|Bischof|Direktor|Präsident|Senator|Reichstagsabgeordneter)\b',
    'lb': r'\b(Ambassadeur|Deputéierten|Minister|Kandidat|Konsul|Gouverneur|Bürgermeeschter|Bischof|President|Senator)\b',
}

# Origin markers — moderate signal for at=PROBABLE
_ORIGIN_PATTERNS = {
    'en': r'\b(of|from|born in|native of|originating from|coming from|hailing from)\s+',
    'fr': r'\b(de|du|d\'|né à|née à|originaire de|venant de|venu de)\s+',
    'de': r'\b(von|aus|geboren in|geboren zu|stammend aus|gebürtig aus)\s+',
    'lb': r'\b(vu|vun|gebuer zu|aus)\s+',
}

# Departure markers
_DEPARTURE_PATTERNS = {
    'en': r'\b(left|departed|leaving|departing|fled from)\s+',
    'fr': r'\b(quitta|quitté|quittant|partit de|s\'échappa de|fuyit)\s+',
    'de': r'\b(verließ|verlassen|reiste ab|floh aus|flüchtete aus|flüchtete von)\s+',
    'lb': r'\b(verlooss|fortgaang|geflücht aus)\s+',
}

# Candidacy markers
_CANDIDACY_PATTERNS = {
    'en': r'\b(candidate (?:in|for)|running for|standing in|election in|nominated for|seat (?:of|in))\b',
    'fr': r'\b(candidat (?:dans|de|pour|à)|se présente dans|circonscription de|siège de|élection à|députation de)\b',
    'de': r'\b(Kandidat (?:in|für)|kandidiert (?:in|für)|Wahlkreis|Sitz (?:in|von)|Wahl in)\b',
    'lb': r'\b(Kandidat (?:zu|fir)|kandidatéiert|Wahlkrees|Wal zu)\b',
}

# Temporal "now" markers
_TEMPORAL_NOW_PATTERNS = {
    'en': r'\b(today|yesterday|tomorrow|currently|this morning|this afternoon|this week|now|presently|recently|just (?:arrived|left))\b',
    'fr': r'\b(aujourd\'hui|hier|demain|actuellement|ce matin|cet après-midi|cette semaine|maintenant|récemment|venait d\'arriver)\b',
    'de': r'\b(heute|gestern|morgen|derzeit|aktuell|diesen Morgen|diese Woche|jetzt|nun|kürzlich|soeben (?:angekommen|verlassen))\b',
    'lb': r'\b(haut|gëschter|moien|aktuell|dës Woch|elo|viru kuerzem)\b',
}


def _extract_local_context(text, pers_mentions, loc_mentions, window_chars=200):
    """Extract local context around person and place mentions.

    Returns text within `window_chars` chars of either mention.
    """
    if not text or not pers_mentions or not loc_mentions:
        return text or ""

    text_lower = text.lower()
    spans = []

    # Find all mention positions
    for mention in (pers_mentions + loc_mentions):
        if not mention: continue
        m_lower = mention.lower()
        start = 0
        while True:
            idx = text_lower.find(m_lower, start)
            if idx == -1: break
            spans.append((max(0, idx - window_chars), min(len(text), idx + len(mention) + window_chars)))
            start = idx + len(mention)

    if not spans:
        return text[:1000]  # fallback

    # Merge overlapping spans
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return " ... ".join(text[s:e] for s, e in merged)


def build_text_pattern_features(doc, pair):
    """Detect linguistic patterns in local context around (person, place) mention.

    Returns 6-dim binary vector:
      [0] action_flag    — locative action verb near pair
      [1] role_flag      — role/position marker (X of [place])
      [2] origin_flag    — origin/birth marker
      [3] temporal_flag  — "now/today/yesterday" markers
      [4] departure_flag — "left/departed" markers (at=TRUE but isAt=FALSE)
      [5] candidacy_flag — "candidate/election" markers (PROBABLE)
    """
    feats = [0.0] * FOL_FEATURE_DIM

    # Get language; default to English regex
    lang = (doc.get('language') or 'en').lower()[:2]
    if lang not in _ACTION_PATTERNS:
        lang = 'en'

    # Extract context around mentions
    pers_mentions = pair.get('pers_mentions_list', []) or []
    loc_mentions = pair.get('loc_mentions_list', []) or []
    text = doc.get('text', '') or ''
    context = _extract_local_context(text, pers_mentions, loc_mentions, window_chars=200)
    if not context:
        return feats

    # Apply patterns
    try:
        if re.search(_ACTION_PATTERNS[lang], context, re.IGNORECASE):
            feats[0] = 1.0
        if re.search(_ROLE_PATTERNS[lang], context, re.IGNORECASE):
            feats[1] = 1.0
        if re.search(_ORIGIN_PATTERNS[lang], context, re.IGNORECASE):
            feats[2] = 1.0
        if re.search(_TEMPORAL_NOW_PATTERNS[lang], context, re.IGNORECASE):
            feats[3] = 1.0
        if re.search(_DEPARTURE_PATTERNS[lang], context, re.IGNORECASE):
            feats[4] = 1.0
        if re.search(_CANDIDACY_PATTERNS[lang], context, re.IGNORECASE):
            feats[5] = 1.0
    except re.error:
        pass

    return feats


# ============================================================
# Input text builder
# ============================================================

def build_input_text(doc, pair, window=400):
    """Build a focused input string for the encoder.

    Format:
      Language: <lang>
      Date: <article_date>
      Person: <person mentions>
      Place: <place mentions>
      Context: <window of text around mentions>
    """
    pers_list = pair.get("pers_mentions_list", []) or []
    loc_list = pair.get("loc_mentions_list", []) or []
    pers_str = " | ".join(pers_list[:3])
    loc_str = " | ".join(loc_list[:3])

    text = doc.get("text", "")
    context = get_pair_context(text, pers_list, loc_list, window=window)

    lang = doc.get("language", "?")
    date = doc.get("date", "?")

    return (
        f"language={lang} date={date} "
        f"person={pers_str} place={loc_str} "
        f"context: {context}"
    )


# ============================================================
# Dataset
# ============================================================

class HipeDataset(_TorchDataset):
    """Pytorch Dataset for HIPE-2026 person-place relations.

    Each example is a single (document, pair) instance.

    Args:
        files: list of jsonl paths
        tokenizer: HuggingFace tokenizer
        max_length: tokenizer max_length
        kg_pairs / kg_persons / kg_locations: dict caches from kg_reasoner
        has_labels: True for train/dev (gold present), False for test
        window: text window around mentions
    """

    def __init__(self, files, tokenizer, max_length=320,
                 kg_pairs=None, kg_persons=None, kg_locations=None,
                 has_labels=True, window=400):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kg_pairs = kg_pairs or {}
        self.kg_persons = kg_persons or {}
        self.kg_locations = kg_locations or {}
        self.has_labels = has_labels
        self.window = window
        self.examples = []  # list of dicts ready for __getitem__

        for path in files:
            self._load_file(path)

    def _load_file(self, path):
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                doc = json.loads(line)
                doc_id = doc.get("document_id")
                article_date = doc.get("date", "")
                for pair_idx, pair in enumerate(doc.get("sampled_pairs", [])):
                    pq = pair.get("pers_wikidata_QID") or ""
                    lq = pair.get("loc_wikidata_QID") or ""

                    text_input = build_input_text(doc, pair, window=self.window)
                    kg_feats = build_kg_features(pq, lq, article_date,
                                                  self.kg_pairs, self.kg_persons, self.kg_locations)
                    fol_feats = build_text_pattern_features(doc, pair)

                    if self.has_labels:
                        at_str = pair.get("at")
                        is_str = pair.get("isAt")
                        at_label = AT_LABEL2ID.get(at_str, -1)
                        is_label = ISAT_LABEL2ID.get(is_str, -1)
                    else:
                        at_label = -1
                        is_label = -1

                    self.examples.append({
                        "doc_id": doc_id,
                        "pair_idx": pair_idx,
                        "input_text": text_input,
                        "kg_features": kg_feats,
                        "fol_features": fol_feats,
                        "at_label": at_label,
                        "isAt_label": is_label,
                        "language": doc.get("language", "?"),
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["input_text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,  # collate_fn pads
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "kg_features": ex["kg_features"],
            "fol_features": ex["fol_features"],
            "at_label": ex["at_label"],
            "isAt_label": ex["isAt_label"],
            "doc_id": ex["doc_id"],
            "pair_idx": ex["pair_idx"],
            "language": ex["language"],
        }


def hipe_collate_fn(batch, pad_token_id=1):
    """Collate function: pads input_ids and attention_mask, stacks others."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append(mask + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "kg_features": torch.tensor([x["kg_features"] for x in batch], dtype=torch.float),
        "fol_features": torch.tensor([x["fol_features"] for x in batch], dtype=torch.float),
        "at_labels": torch.tensor([x["at_label"] for x in batch], dtype=torch.long),
        "isAt_labels": torch.tensor([x["isAt_label"] for x in batch], dtype=torch.long),
        "doc_ids": [x["doc_id"] for x in batch],
        "pair_idxs": [x["pair_idx"] for x in batch],
        "languages": [x["language"] for x in batch],
    }


# ============================================================
# Cache loader
# ============================================================

def load_kg_caches(kg_facts_path):
    """Load the 3 KG cache files produced by kg_reasoner.py."""
    kg_pairs, kg_persons, kg_locations = {}, {}, {}
    for path, target in [
        (kg_facts_path, kg_pairs),
        (kg_facts_path.replace(".jsonl", "_persons.jsonl"), kg_persons),
        (kg_facts_path.replace(".jsonl", "_locations.jsonl"), kg_locations),
    ]:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    key = row.get("_key")
                    if key:
                        target[key] = row
    return kg_pairs, kg_persons, kg_locations


if __name__ == "__main__":
    # Smoke test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hipe_dataset.py <jsonl_file> [kg_facts.jsonl]")
        sys.exit(1)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

    kg_pairs, kg_persons, kg_locations = ({}, {}, {})
    if len(sys.argv) >= 3:
        kg_pairs, kg_persons, kg_locations = load_kg_caches(sys.argv[2])
        print(f"KG: {len(kg_pairs)} pairs, {len(kg_persons)} persons, {len(kg_locations)} locations")

    ds = HipeDataset([sys.argv[1]], tok,
                      kg_pairs=kg_pairs, kg_persons=kg_persons, kg_locations=kg_locations)
    print(f"Dataset size: {len(ds)}")
    print(f"First example:")
    ex = ds[0]
    print(f"  input_ids[:20]: {ex['input_ids'][:20]}")
    print(f"  kg_features: {ex['kg_features']}")
    print(f"  at_label: {ex['at_label']} ({AT_LABELS[ex['at_label']] if ex['at_label']>=0 else 'unlabeled'})")
    print(f"  isAt_label: {ex['isAt_label']} ({ISAT_LABELS[ex['isAt_label']] if ex['isAt_label']>=0 else 'unlabeled'})")
    print(f"  language: {ex['language']}")

    # Class distribution
    from collections import Counter
    at_counts = Counter(ex["at_label"] for ex in ds.examples)
    is_counts = Counter(ex["isAt_label"] for ex in ds.examples)
    print(f"\n'at' distribution:")
    for i, l in enumerate(AT_LABELS):
        print(f"  {l}: {at_counts.get(i, 0)}")
    print(f"\n'isAt' distribution:")
    for i, l in enumerate(ISAT_LABELS):
        print(f"  {l}: {is_counts.get(i, 0)}")
