#!/usr/bin/env python3
"""Download a pinned public Kaggle ZIP and freeze a chronological study."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import re
import urllib.request
import zipfile

from .paths import RAW_ARCHIVE
from .study import LABELS, PRODUCTS, digest, object_hash, read_jsonl, write_jsonl, request_for

URL = "https://www.kaggle.com/api/v1/datasets/download/selener/consumer-complaint-database?datasetVersionNumber=1"
SOURCE = "https://www.kaggle.com/datasets/selener/consumer-complaint-database"


def phase(date):
    if "2015-01-01" <= date < "2016-07-01":
        return "train"
    if "2016-07-01" <= date < "2016-10-01":
        return "dev"
    if "2016-10-01" <= date < "2017-01-01":
        return "test"
    return None


# Opt-in fresh-window support. The default prepare() path above is unchanged.
# CFPB changed its product taxonomy effective 24 April 2017; complaints received
# before that date keep the product names they were submitted under (CFPB,
# "Summary of Product and Sub-product Changes", April 2017). Option A therefore
# uses 2017-01-01 through 2017-04-23 inclusive. Option B is the fallback.
FRESH_WINDOWS = {
    "2017q1": ("2017-01-01", "2017-04-24",
               "CFPB product taxonomy changed effective 2017-04-24; earlier records keep old names"),
    "unused-2016q4": ("2016-10-01", "2017-01-01",
                      "Fallback: 2016 Q4 records whose complaint IDs and texts are absent from the original study"),
}
FRESH_TAXONOMY_SOURCE = "https://files.consumerfinance.gov/f/documents/201704_cfpb_Summary_of_Product_and_Sub-product_Changes.pdf"


def study_exclusions(path):
    records = read_jsonl(path)
    return {r["id"] for r in records}, {r["group_hash"] for r in records}


def prepare_fresh(archive, out, window, exclude_study, n=2000, max_chars=4000, seed_tag=None):
    if window not in FRESH_WINDOWS:
        raise ValueError("Unknown fresh window")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory")
    start, end, rationale = FRESH_WINDOWS[window]
    seed_tag = seed_tag or f"fresh-{window}-20261023"
    study_ids, study_hashes = study_exclusions(exclude_study)
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    groups = {}
    conflicts = set()
    with zipfile.ZipFile(archive) as zipped:
        member = zipped.getinfo("rows.csv")
        with zipped.open(member) as binary:
            reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
            required = {"Date received", "Product", "Consumer complaint narrative", "Complaint ID"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Unexpected dataset schema")
            for row in reader:
                counts["raw_rows"] += 1
                product = row["Product"]
                if product not in PRODUCTS:
                    counts["excluded_product"] += 1
                    continue
                date = datetime.strptime(row["Date received"], "%m/%d/%Y").date().isoformat()
                if not start <= date < end:
                    counts["excluded_date"] += 1
                    continue
                if row["Complaint ID"] in study_ids:
                    counts["excluded_seen_in_study_id"] += 1
                    continue
                text = re.sub(r"\s+", " ", row["Consumer complaint narrative"]).strip()
                if len(text) < 40:
                    counts["missing_or_short_narrative"] += 1
                    continue
                text_used = text[:max_chars]
                group = object_hash(text_used.lower())
                if group in study_hashes:
                    counts["excluded_seen_in_study_text"] += 1
                    continue
                record = {"id": row["Complaint ID"], "date": date, "split": "fresh",
                          "label": PRODUCTS[product], "text": text_used, "group_hash": group,
                          "original_chars": len(text), "truncated": len(text) > max_chars}
                counts["eligible_before_dedup"] += 1
                if group in groups:
                    counts["duplicate_model_input"] += 1
                    old = groups[group]
                    if old["label"] != record["label"]:
                        conflicts.add(group)
                    if (record["date"], int(record["id"])) < (old["date"], int(old["id"])):
                        groups[group] = record
                else:
                    groups[group] = record
    eligible = [r for key, r in groups.items() if key not in conflicts]
    counts["conflicting_input_groups_removed"] = len(conflicts)
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} eligible fresh records, need {n}")
    # Label-blind deterministic sampling with a seed tag distinct from the original study.
    chosen = sorted(eligible, key=lambda r: object_hash([seed_tag, r["id"]]))[:n]
    chosen.sort(key=lambda r: (r["date"], int(r["id"])))
    if len({r["id"] for r in chosen}) != len(chosen):
        raise ValueError("Complaint IDs are not unique")
    write_jsonl(out/"study.jsonl", chosen)
    requests = [{"id": r["id"], "split": "fresh", "request": request_for(r)} for r in chosen]
    write_jsonl(out/"requests.jsonl", requests)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "source_url": SOURCE, "download_url": URL, "dataset_version": 1,
                "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size,
                "csv_uncompressed_bytes": member.file_size, "csv_crc32": f"{member.CRC:08x}",
                "fresh": True, "fresh_window": window, "window_start": start, "window_end_exclusive": end,
                "window_rationale": rationale, "taxonomy_source": FRESH_TAXONOMY_SOURCE,
                "exclude_study": {"sha256": digest(exclude_study), "complaint_ids": len(study_ids),
                                  "group_hashes": len(study_hashes)},
                "seed_tag": seed_tag, "max_chars": max_chars, "labels": LABELS,
                "filter_counts": dict(counts), "available_after_dedup": len(eligible),
                "near_duplicate_detection": False,
                "study_sha256": digest(out/"study.jsonl"), "requests_sha256": digest(out/"requests.jsonl"),
                "splits": {"fresh": {"n": len(chosen), "counts": dict(Counter(r["label"] for r in chosen)),
                                     "start": min(r["date"] for r in chosen), "end": max(r["date"] for r in chosen),
                                     "truncated": sum(r["truncated"] for r in chosen)}}}
    (out/"data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"fresh_window": window, "selected": len(chosen), "available_after_dedup": len(eligible),
                      "output": str(out)}, indent=2))


def download(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    request = urllib.request.Request(URL, headers={"User-Agent": "reproducible-feedback-study/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as handle:
        if response.status != 200:
            raise ValueError("Kaggle did not return HTTP 200")
        for block in iter(lambda: response.read(1024*1024), b""):
            handle.write(block)
    with zipfile.ZipFile(partial) as archive:
        if "rows.csv" not in archive.namelist():
            raise ValueError("Expected rows.csv in the pinned Kaggle archive")
    partial.replace(destination)


def prepare(archive, out, train=12000, dev=1000, test=2000, max_chars=4000, seed=20261022):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    groups = {}
    conflicts = set()
    with zipfile.ZipFile(archive) as zipped:
        member = zipped.getinfo("rows.csv")
        with zipped.open(member) as binary:
            reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
            required = {"Date received", "Product", "Consumer complaint narrative", "Complaint ID"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Unexpected dataset schema")
            for row in reader:
                counts["raw_rows"] += 1
                product = row["Product"]
                if product not in PRODUCTS:
                    counts["excluded_product"] += 1
                    continue
                date = datetime.strptime(row["Date received"], "%m/%d/%Y").date().isoformat()
                split = phase(date)
                if split is None:
                    counts["excluded_date"] += 1
                    continue
                text = re.sub(r"\s+", " ", row["Consumer complaint narrative"]).strip()
                if len(text) < 40:
                    counts["missing_or_short_narrative"] += 1
                    continue
                text_used = text[:max_chars]
                group = object_hash(text_used.lower())
                record = {"id": row["Complaint ID"], "date": date, "split": split,
                          "label": PRODUCTS[product], "text": text_used, "group_hash": group,
                          "original_chars": len(text), "truncated": len(text)>max_chars}
                counts["eligible_before_dedup"] += 1
                if group in groups:
                    counts["duplicate_model_input"] += 1
                    old = groups[group]
                    if old["label"] != record["label"]:
                        conflicts.add(group)
                    if (record["date"], int(record["id"])) < (old["date"], int(old["id"])):
                        groups[group] = record
                else:
                    groups[group] = record
    eligible = [r for key,r in groups.items() if key not in conflicts]
    counts["conflicting_input_groups_removed"] = len(conflicts)
    selected = []
    availability = {}
    for split, cap in (("train", train), ("dev", dev), ("test", test)):
        cell = [r for r in eligible if r["split"]==split]
        availability[split] = len(cell)
        if len(cell) < cap:
            raise ValueError(f"Only {len(cell)} eligible {split} records, need {cap}")
        # Label-blind deterministic pseudo-random sampling. No equal-class quota.
        chosen = sorted(cell, key=lambda r: object_hash([seed, r["id"]]))[:cap]
        chosen.sort(key=lambda r: (r["date"], int(r["id"])))
        selected.extend(chosen)
    if len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Complaint IDs are not unique")
    write_jsonl(out/"study.jsonl", selected)
    requests = [{"id":r["id"],"split":r["split"],"request":request_for(r)} for r in selected if r["split"]!="train"]
    write_jsonl(out/"requests.jsonl", requests)
    manifest = {"created_utc":datetime.now(timezone.utc).isoformat(), "python":platform.python_version(),
                "source_url":SOURCE,"download_url":URL,"dataset_version":1,
                "archive_sha256":digest(archive),"archive_bytes":archive.stat().st_size,
                "csv_uncompressed_bytes":member.file_size,"csv_crc32":f"{member.CRC:08x}",
                "study_sha256":digest(out/"study.jsonl"),"requests_sha256":digest(out/"requests.jsonl"),
                "seed":seed,"max_chars":max_chars,"labels":LABELS,"filter_counts":dict(counts),
                "available_after_dedup":availability,"near_duplicate_detection":False,
                "splits":{s:{"n":sum(r["split"]==s for r in selected),
                             "counts":dict(Counter(r["label"] for r in selected if r["split"]==s)),
                             "start":min(r["date"] for r in selected if r["split"]==s),
                             "end":max(r["date"] for r in selected if r["split"]==s),
                             "truncated":sum(r["truncated"] for r in selected if r["split"]==s)} for s in ("train","dev","test")}}
    (out/"data-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"selected":len(selected),"split_sizes":{s:v['n'] for s,v in manifest['splits'].items()},"output":str(out)},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=RAW_ARCHIVE)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--train",type=int,default=12000)
    parser.add_argument("--dev",type=int,default=1000)
    parser.add_argument("--test",type=int,default=2000)
    parser.add_argument("--max-chars",type=int,default=4000)
    parser.add_argument("--fresh",choices=sorted(FRESH_WINDOWS),help="Build an unseen fresh set instead of the original study")
    parser.add_argument("--exclude-study",type=Path,help="Original study.jsonl; its complaint IDs and text hashes are excluded (required with --fresh)")
    parser.add_argument("--fresh-n",type=int,default=2000)
    args=parser.parse_args()
    if min(args.train,args.dev,args.test) < 1 or not 100 <= args.max_chars <= 10000:
        parser.error("Positive split sizes and 100–10,000 max characters required")
    if args.fresh:
        if args.exclude_study is None or not args.exclude_study.is_file():
            parser.error("--fresh requires --exclude-study PATH to an existing study.jsonl")
        if args.fresh_n < 1:
            parser.error("--fresh-n must be positive")
        if not args.archive.exists():
            download(args.archive)
        prepare_fresh(args.archive,args.out,args.fresh,args.exclude_study,args.fresh_n,args.max_chars)
        return
    if not args.archive.exists():
        download(args.archive)
    prepare(args.archive,args.out,args.train,args.dev,args.test,args.max_chars)


if __name__=="__main__":
    main()
