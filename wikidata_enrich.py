#!/usr/bin/env python3
"""
HIPE-2026 Wikidata Enrichment
==============================
Fetches structured knowledge for all person/location QIDs from training data.

For persons: birth/death dates, birth/death places, nationality, occupation
For locations: country, coordinates, type (city/country/region)

Usage:
    python wikidata_enrich.py --data_dir all_train --output wikidata_cache.json
"""

import json
import os
import sys
import time
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict


WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "HIPE2026-Research/1.0 (academic research)"}


def sparql_query(query, retries=3):
    """Execute SPARQL query against Wikidata."""
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    url = f"{WIKIDATA_SPARQL}?{params}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  SPARQL attempt {attempt+1} failed: {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    return None


def fetch_person_info(qids):
    """Fetch person info in batches."""
    results = {}
    batch_size = 50  # Wikidata rate limit friendly

    for i in range(0, len(qids), batch_size):
        batch = qids[i:i+batch_size]
        values = " ".join(f"wd:{q}" for q in batch)
        print(f"  Fetching persons {i+1}-{min(i+batch_size, len(qids))}/{len(qids)}...", flush=True)

        query = f"""
        SELECT ?person ?personLabel
               ?birthDate ?deathDate
               ?birthPlaceLabel ?birthPlace
               ?deathPlaceLabel ?deathPlace
               ?nationalityLabel ?nationality
               ?occupationLabel
        WHERE {{
            VALUES ?person {{ {values} }}
            OPTIONAL {{ ?person wdt:P569 ?birthDate. }}
            OPTIONAL {{ ?person wdt:P570 ?deathDate. }}
            OPTIONAL {{ ?person wdt:P19 ?birthPlace. ?birthPlace rdfs:label ?birthPlaceLabel. FILTER(LANG(?birthPlaceLabel) = "en") }}
            OPTIONAL {{ ?person wdt:P20 ?deathPlace. ?deathPlace rdfs:label ?deathPlaceLabel. FILTER(LANG(?deathPlaceLabel) = "en") }}
            OPTIONAL {{ ?person wdt:P27 ?nationality. ?nationality rdfs:label ?nationalityLabel. FILTER(LANG(?nationalityLabel) = "en") }}
            OPTIONAL {{ ?person wdt:P106 ?occupation. ?occupation rdfs:label ?occupationLabel. FILTER(LANG(?occupationLabel) = "en") }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,de,fr". }}
        }}
        """

        data = sparql_query(query)
        if not data:
            continue

        for row in data["results"]["bindings"]:
            qid = row["person"]["value"].split("/")[-1]
            if qid not in results:
                results[qid] = {
                    "qid": qid,
                    "label": row.get("personLabel", {}).get("value", ""),
                    "type": "person",
                    "birth_date": None,
                    "death_date": None,
                    "birth_place": None,
                    "birth_place_qid": None,
                    "death_place": None,
                    "death_place_qid": None,
                    "nationalities": [],
                    "occupations": [],
                }

            r = results[qid]
            if "birthDate" in row:
                r["birth_date"] = row["birthDate"]["value"][:10]
            if "deathDate" in row:
                r["death_date"] = row["deathDate"]["value"][:10]
            if "birthPlaceLabel" in row:
                r["birth_place"] = row["birthPlaceLabel"]["value"]
                r["birth_place_qid"] = row.get("birthPlace", {}).get("value", "").split("/")[-1]
            if "deathPlaceLabel" in row:
                r["death_place"] = row["deathPlaceLabel"]["value"]
                r["death_place_qid"] = row.get("deathPlace", {}).get("value", "").split("/")[-1]
            if "nationalityLabel" in row:
                nat = row["nationalityLabel"]["value"]
                if nat not in r["nationalities"]:
                    r["nationalities"].append(nat)
            if "occupationLabel" in row:
                occ = row["occupationLabel"]["value"]
                if occ not in r["occupations"]:
                    r["occupations"].append(occ)

        time.sleep(2)  # Be nice to Wikidata

    return results


def fetch_location_info(qids):
    """Fetch location info in batches."""
    results = {}
    batch_size = 50

    for i in range(0, len(qids), batch_size):
        batch = qids[i:i+batch_size]
        values = " ".join(f"wd:{q}" for q in batch)
        print(f"  Fetching locations {i+1}-{min(i+batch_size, len(qids))}/{len(qids)}...", flush=True)

        query = f"""
        SELECT ?location ?locationLabel
               ?countryLabel ?country
               ?coord
               ?instanceLabel
        WHERE {{
            VALUES ?location {{ {values} }}
            OPTIONAL {{ ?location wdt:P17 ?country. ?country rdfs:label ?countryLabel. FILTER(LANG(?countryLabel) = "en") }}
            OPTIONAL {{ ?location wdt:P625 ?coord. }}
            OPTIONAL {{ ?location wdt:P31 ?instance. ?instance rdfs:label ?instanceLabel. FILTER(LANG(?instanceLabel) = "en") }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,de,fr". }}
        }}
        """

        data = sparql_query(query)
        if not data:
            continue

        for row in data["results"]["bindings"]:
            qid = row["location"]["value"].split("/")[-1]
            if qid not in results:
                results[qid] = {
                    "qid": qid,
                    "label": row.get("locationLabel", {}).get("value", ""),
                    "type": "location",
                    "country": None,
                    "country_qid": None,
                    "latitude": None,
                    "longitude": None,
                    "instance_types": [],
                }

            r = results[qid]
            if "countryLabel" in row:
                r["country"] = row["countryLabel"]["value"]
                r["country_qid"] = row.get("country", {}).get("value", "").split("/")[-1]
            if "coord" in row:
                coord = row["coord"]["value"]
                # Parse "Point(lon lat)"
                try:
                    parts = coord.replace("Point(", "").replace(")", "").split()
                    r["longitude"] = float(parts[0])
                    r["latitude"] = float(parts[1])
                except:
                    pass
            if "instanceLabel" in row:
                inst = row["instanceLabel"]["value"]
                if inst not in r["instance_types"]:
                    r["instance_types"].append(inst)

        time.sleep(2)

    return results


def extract_qids(data_dir):
    """Extract all unique QIDs from JSONL files."""
    pers_qids = set()
    loc_qids = set()

    for fpath in Path(data_dir).glob("*.jsonl"):
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                for pair in doc["sampled_pairs"]:
                    pq = pair.get("pers_wikidata_QID")
                    lq = pair.get("loc_wikidata_QID")
                    if pq and pq.startswith("Q"):
                        pers_qids.add(pq)
                    if lq and lq.startswith("Q"):
                        loc_qids.add(lq)

    return sorted(pers_qids), sorted(loc_qids)


def compute_pair_features(pers_info, loc_info, doc_date):
    """Compute features for a (person, location) pair."""
    features = {
        # Person features
        "person_has_wikidata": pers_info is not None,
        "person_is_dead": False,
        "person_died_before_article": False,
        "person_born_at_location": False,
        "person_died_at_location": False,
        "person_nationality_matches_location": False,
        "person_is_politician": False,
        "person_is_military": False,
        "person_is_religious": False,
        # Location features
        "location_has_wikidata": loc_info is not None,
        "location_is_country": False,
        "location_is_city": False,
        # Pair features
        "same_country": False,
        "geo_distance_km": -1,
    }

    if pers_info:
        # Death check
        if pers_info.get("death_date"):
            features["person_is_dead"] = True
            try:
                if pers_info["death_date"] < doc_date:
                    features["person_died_before_article"] = True
            except:
                pass

        # Birth/death place match
        if loc_info:
            loc_qid = loc_info["qid"]
            if pers_info.get("birth_place_qid") == loc_qid:
                features["person_born_at_location"] = True
            if pers_info.get("death_place_qid") == loc_qid:
                features["person_died_at_location"] = True

            # Nationality matches location country
            if loc_info.get("country"):
                for nat in pers_info.get("nationalities", []):
                    if nat.lower() in loc_info["country"].lower() or loc_info["country"].lower() in nat.lower():
                        features["person_nationality_matches_location"] = True

            # Same country
            if loc_info.get("country_qid") and pers_info.get("birth_place_qid"):
                # This is approximate — would need birth place's country
                pass

        # Occupation classification
        occs = " ".join(pers_info.get("occupations", [])).lower()
        features["person_is_politician"] = any(w in occs for w in ["politician", "statesman", "president", "minister", "diplomat", "senator", "governor"])
        features["person_is_military"] = any(w in occs for w in ["military", "officer", "general", "admiral", "soldier", "commander"])
        features["person_is_religious"] = any(w in occs for w in ["priest", "bishop", "archbishop", "pope", "cardinal", "rabbi", "pastor"])

    if loc_info:
        types = " ".join(loc_info.get("instance_types", [])).lower()
        features["location_is_country"] = any(w in types for w in ["country", "sovereign state", "state"])
        features["location_is_city"] = any(w in types for w in ["city", "town", "municipality", "commune", "village"])

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", nargs="+", default=["all_train", "sandbox_dev"],
                        help="Directories to scan for QIDs")
    parser.add_argument("--output", default="wikidata_cache.json")
    parser.add_argument("--features_output", default="wikidata_features_example.json")
    args = parser.parse_args()

    # Collect QIDs from all directories
    all_pers_qids = set()
    all_loc_qids = set()
    for d in args.data_dir:
        if os.path.isdir(d):
            pq, lq = extract_qids(d)
            all_pers_qids.update(pq)
            all_loc_qids.update(lq)
            print(f"{d}: {len(pq)} person QIDs, {len(lq)} location QIDs")

    all_pers_qids = sorted(all_pers_qids)
    all_loc_qids = sorted(all_loc_qids)
    print(f"\nTotal unique: {len(all_pers_qids)} persons, {len(all_loc_qids)} locations")

    # Fetch from Wikidata
    print("\n--- Fetching person info ---")
    person_data = fetch_person_info(all_pers_qids)
    print(f"Got info for {len(person_data)} persons")

    print("\n--- Fetching location info ---")
    location_data = fetch_location_info(all_loc_qids)
    print(f"Got info for {len(location_data)} locations")

    # Combine and save
    cache = {
        "persons": person_data,
        "locations": location_data,
    }

    with open(args.output, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"\nSaved cache to {args.output}")

    # Print some examples
    print("\n--- Sample person entries ---")
    for qid in list(person_data.keys())[:3]:
        p = person_data[qid]
        print(f"  {qid}: {p['label']} | born={p['birth_date']} died={p['death_date']} | "
              f"nat={p['nationalities'][:2]} | occ={p['occupations'][:2]}")

    print("\n--- Sample location entries ---")
    for qid in list(location_data.keys())[:3]:
        l = location_data[qid]
        print(f"  {qid}: {l['label']} | country={l['country']} | "
              f"types={l['instance_types'][:2]} | coord=({l['latitude']},{l['longitude']})")

    # Generate example features for first doc
    print("\n--- Example pair features ---")
    for d in args.data_dir:
        if not os.path.isdir(d):
            continue
        for fpath in sorted(Path(d).glob("*.jsonl")):
            with open(fpath) as f:
                doc = json.loads(f.readline())
            for pair in doc["sampled_pairs"][:2]:
                pq = pair.get("pers_wikidata_QID", "")
                lq = pair.get("loc_wikidata_QID", "")
                pi = person_data.get(pq)
                li = location_data.get(lq)
                feats = compute_pair_features(pi, li, doc.get("date", ""))
                print(f"\n  {pair['pers_mentions_list'][0]} × {pair['loc_mentions_list'][0]}")
                print(f"  Person: {pi['label'] if pi else 'NO QID'}")
                print(f"  Location: {li['label'] if li else 'NO QID'}")
                print(f"  Gold: at={pair['at']} isAt={pair['isAt']}")
                active_feats = {k: v for k, v in feats.items() if v not in (False, -1, None)}
                print(f"  Active features: {active_feats}")
            break
        break


if __name__ == "__main__":
    main()
