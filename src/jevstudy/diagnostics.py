#!/usr/bin/env python3
"""Read-only diagnostics for a completed run.

Writes <run>/diagnostics.json with paired McNemar tests (test and dev), dev-set
metrics and share errors, the largest off-diagonal confusion cell per
probabilistic method against its review threshold, Jev debt_collection misroute
term shares, and reliability bins. No model calls; run files are only read.

Usage: python -m jevstudy.diagnostics --run results/main
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .metrics import confusion, reliability, scalar_metrics
from .study import LABELS, digest, object_hash, read_jsonl, validate_answer

ROOT = Path(__file__).resolve().parent  # package directory
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}
TERMS = {
    "credit_report": r"credit report",
    "bureau": r"\b(equifax|experian|transunion|trans union|credit bureau|credit reporting agenc\w*|bureaus?)\b",
    "collector": r"\b(collector|collection agency|collections? agency|debt collector)\b",
    "validation": r"\b(fdcpa|validation|validate|verify the debt|debt validation)\b",
    "dispute": r"\bdisput",
}
INPUT_FILES = ["data-manifest.json", "local-manifest.json", "study.jsonl", "requests.jsonl",
               "local-predictions.jsonl", "jev-responses.jsonl", "review-policy.json"]


def load_run(folder):
    data = json.loads((folder / "data-manifest.json").read_text(encoding="utf-8"))
    if digest(folder / "study.jsonl") != data["study_sha256"] or digest(folder / "requests.jsonl") != data["requests_sha256"]:
        raise ValueError("Frozen study/request hash mismatch")
    local = json.loads((folder / "local-manifest.json").read_text(encoding="utf-8"))
    if digest(folder / "local-predictions.jsonl") != local["predictions_sha256"] or local["study_sha256"] != data["study_sha256"]:
        raise ValueError("Local predictions do not match frozen study")
    rows = read_jsonl(folder / "study.jsonl")
    by_id = {r["id"]: r for r in rows}
    methods = {"keyword": {}, "tfidf": {}}
    for r in read_jsonl(folder / "local-predictions.jsonl"):
        if r["id"] in methods[r["method"]] or r["id"] not in by_id or r["input_hash"] != object_hash(by_id[r["id"]]["text"]):
            raise ValueError("Invalid/duplicate local prediction or input hash")
        methods[r["method"]][r["id"]] = {"label": r["label"], "probabilities": r["probabilities"]}
    requests = {r["id"]: r for r in read_jsonl(folder / "requests.jsonl")}
    jev = {}
    for r in read_jsonl(folder / "jev-responses.jsonl"):
        if r["status"] != "ok":
            continue
        if r["id"] in jev or r["id"] not in requests or r["request_hash"] != object_hash(requests[r["id"]]["request"]):
            raise ValueError("Duplicate or mismatched Jev response")
        answer = validate_answer(r["response"])
        jev[r["id"]] = {"label": answer["label"], "probabilities": answer["probabilities"]}
    methods["jev"] = jev
    evaluated = {r["id"] for r in rows if r["split"] in ("dev", "test")}
    for name, predictions in methods.items():
        if set(predictions) != evaluated:
            raise ValueError(f"Incomplete predictions for {name}")
    policy = json.loads((folder / "review-policy.json").read_text(encoding="utf-8"))
    thresholds = {name: policy[name]["threshold"] for name in ("tfidf", "jev")}
    return rows, methods, thresholds


def arrays(rows, predictions):
    truth = np.array([LABEL_INDEX[r["label"]] for r in rows])
    pred = np.array([LABEL_INDEX[predictions[r["id"]]["label"]] for r in rows])
    raw = [predictions[r["id"]]["probabilities"] for r in rows]
    probs = np.array(raw, dtype=float) if all(p is not None for p in raw) else None
    return truth, pred, probs


def mcnemar_counts(truth, jev_pred, tfidf_pred):
    jev_ok = jev_pred == truth
    tfidf_ok = tfidf_pred == truth
    b = int(np.sum(jev_ok & ~tfidf_ok))
    c = int(np.sum(~jev_ok & tfidf_ok))
    discordant = b + c
    return {
        "n": int(len(truth)),
        "jev_accuracy": float(jev_ok.mean()),
        "tfidf_accuracy": float(tfidf_ok.mean()),
        "jev_minus_tfidf_accuracy_pp": float((jev_ok.mean() - tfidf_ok.mean()) * 100),
        "jev_right_tfidf_wrong": b,
        "tfidf_right_jev_wrong": c,
        "both_right": int(np.sum(jev_ok & tfidf_ok)),
        "both_wrong": int(np.sum(~jev_ok & ~tfidf_ok)),
        "exact_two_sided_p": float(binomtest(b, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0,
        "continuity_corrected_z": float((abs(b - c) - 1) / np.sqrt(discordant)) if discordant else None,
        "continuity_corrected_chi2": float((abs(b - c) - 1) ** 2 / discordant) if discordant else None,
        "more_accurate_on_discordant_pairs": "jev" if b > c else "tfidf" if c > b else "neither",
    }


def summarise(rows, predictions):
    truth, pred, probs = arrays(rows, predictions)
    m = scalar_metrics(truth, pred, probs)
    return {
        "n": m["n"],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "per_class_f1": dict(zip(LABELS, m["per_class_f1"])),
        "share_error_pp_by_label": dict(zip(LABELS, m["share_error_pp"])),
        "max_abs_share_error_pp": m["max_abs_share_error_pp"],
        "max_abs_share_error_label": LABELS[int(np.argmax(np.abs(m["share_error_pp"])))],
        "brier": m.get("brier"),
    }


def largest_confusion_cell(truth, pred, probs, threshold):
    off = confusion(truth, pred).astype(float)
    np.fill_diagonal(off, -1)
    true_i, pred_i = np.unravel_index(int(np.argmax(off)), off.shape)
    cell = (truth == true_i) & (pred == pred_i)
    cleared = cell & (probs.max(axis=1) >= threshold)
    return {
        "true": LABELS[true_i],
        "predicted": LABELS[pred_i],
        "count": int(cell.sum()),
        "threshold": float(threshold),
        "cleared_threshold": int(cleared.sum()),
        "mean_true_label_probability": float(probs[cell, true_i].mean()) if cell.any() else None,
        "mean_true_label_probability_cleared": float(probs[cleared, true_i].mean()) if cleared.any() else None,
    }


def term_shares(test_rows, jev_labels):
    misrouted = [r for r in test_rows if r["label"] == "debt_collection" and jev_labels[r["id"]] == "credit_reporting"]
    correct = [r for r in test_rows if r["label"] == "debt_collection" and jev_labels[r["id"]] == "debt_collection"]
    result = {"misrouted_n": len(misrouted), "correct_n": len(correct), "patterns": {}}
    for name, pattern in TERMS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        mis = sum(1 for r in misrouted if rx.search(r["text"]))
        cor = sum(1 for r in correct if rx.search(r["text"]))
        result["patterns"][name] = {
            "regex": pattern,
            "misrouted_matches": mis,
            "misrouted_share": mis / len(misrouted) if misrouted else None,
            "correct_matches": cor,
            "correct_share": cor / len(correct) if correct else None,
        }
    return result


def reliability_summary(truth, pred, probs):
    top = probs.max(axis=1)
    return {
        "n": int(len(truth)),
        "bins": reliability(truth, pred, probs),
        "median_top_probability": float(np.median(top)),
        "share_top_probability_at_least_0_9": float(np.mean(top >= 0.9)),
    }


def build(folder):
    rows, methods, thresholds = load_run(folder)
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    mcnemar = {}
    for name, split_rows in (("test", test), ("dev", dev)):
        truth, tfidf_pred, _ = arrays(split_rows, methods["tfidf"])
        _, jev_pred, _ = arrays(split_rows, methods["jev"])
        mcnemar[name] = mcnemar_counts(truth, jev_pred, tfidf_pred)
    dev_metrics = {name: summarise(dev, methods[name]) for name in ("keyword", "tfidf", "jev")}
    cells = {}
    reliability_out = {}
    for name in ("tfidf", "jev"):
        truth, pred, probs = arrays(test, methods[name])
        cells[name] = largest_confusion_cell(truth, pred, probs, thresholds[name])
        reliability_out[name] = reliability_summary(truth, pred, probs)
    jev_labels = {rid: pred["label"] for rid, pred in methods["jev"].items()}
    # Extra traceability numbers.
    raw_sums = []
    for r in read_jsonl(folder / "jev-responses.jsonl"):
        if r["status"] == "ok":
            raw_sums.append(sum(r["response"]["answers"]["product"]["probabilities"].values()))
    tfidf_labels = {rid: pred["label"] for rid, pred in methods["tfidf"].items()}
    misroutes = [r["id"] for r in test if r["label"] == "debt_collection" and jev_labels[r["id"]] == "credit_reporting"]
    jev_top = [max(methods["jev"][r["id"]]["probabilities"]) for r in test]
    extras = {
        "jev_ok_responses": len(raw_sums),
        "jev_raw_probability_sums_equal_one": sum(abs(x - 1) < 1e-9 for x in raw_sums),
        "jev_debt_to_reporting_misroutes": len(misroutes),
        "of_which_tfidf_correct": sum(tfidf_labels[i] == "debt_collection" for i in misroutes),
        "jev_test_share_top_prob_ge_0_99": sum(p >= 0.99 for p in jev_top) / len(jev_top),
        "jev_test_share_top_prob_eq_1": sum(p >= 1 - 1e-9 for p in jev_top) / len(jev_top),
        "note": "Share at top probability ~1 bounds how low Jev's review-coverage curve can go.",
    }
    report = {
        "extras": extras,
        "generated_by": "diagnostics.py (read-only analysis of completed run files; no model calls)",
        "inputs_sha256": {name: digest(folder / name) for name in INPUT_FILES},
        "diagnostics_py_sha256": digest(ROOT / "diagnostics.py"),
        "n": {"dev": len(dev), "test": len(test)},
        "mcnemar_jev_vs_tfidf": mcnemar,
        "dev_metrics": dev_metrics,
        "review_thresholds": thresholds,
        "test_largest_confusion_cell": cells,
        "jev_debt_collection_misroute_terms": term_shares(test, jev_labels),
        "test_reliability": reliability_out,
    }
    (folder / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("runs/my-run"))
    args = parser.parse_args()
    folder = args.run.resolve()
    build(folder)
    print(f"Diagnostics: {folder / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
