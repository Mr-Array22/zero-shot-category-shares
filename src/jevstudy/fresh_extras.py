#!/usr/bin/env python3
"""Exploratory (not pre-registered) fresh-set comparisons. No model calls.

Writes results/fresh-2017/fresh-extras.json:
- Jev v2 vs TF-IDF: exact McNemar on accuracy, paired bootstrap of max abs share error.
- ACC share correction (confusion matrix from the ORIGINAL dev split) applied to fresh Jev v1,
  and (dev-v2 confusion) to fresh Jev v2.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from . import revision, study
from .study import LABELS, digest
from .paths import RESULTS_DIR

IX = {l: i for i, l in enumerate(LABELS)}


def jev_labels(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d["status"] == "ok":
            out[d["id"]] = IX[study.validate_answer(d["response"])["label"]]
    return out


def max_share(t, p):
    return float(100 * np.abs(np.bincount(p, minlength=5) - np.bincount(t, minlength=5)).max() / len(t))


def acc_correct(dev_truth, dev_pred, test_pred):
    m = np.zeros((5, 5))
    for t, p in zip(dev_truth, dev_pred):
        m[p, t] += 1
    m /= m.sum(0, keepdims=True)
    q = np.bincount(test_pred, minlength=5) / len(test_pred)
    return np.linalg.solve(m, q)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=RESULTS_DIR / "main")
    ap.add_argument("--fresh", type=Path, default=RESULTS_DIR / "fresh-2017")
    ap.add_argument("--dev-v2", type=Path, default=RESULTS_DIR / "dev-v2")
    a = ap.parse_args()
    src = [json.loads(l) for l in open(a.source / "study.jsonl", encoding="utf-8")]
    by_id = {r["id"]: r for r in src}
    fresh = [json.loads(l) for l in open(a.fresh / "study.jsonl", encoding="utf-8")]
    ids = [r["id"] for r in fresh]
    truth = np.array([IX[r["label"]] for r in fresh])
    pipe, ind = revision.fit_tfidf([r for r in src if r["split"] == "train"])
    tf, _ = revision.predict_tfidf(pipe, ind, fresh)
    tf = np.asarray(tf, dtype=int)
    v1 = jev_labels(a.fresh / "jev-v1" / "jev-responses.jsonl")
    v2 = jev_labels(a.fresh / "jev-v2" / "jev-responses.jsonl")
    j1 = np.array([v1[i] for i in ids]); j2 = np.array([v2[i] for i in ids])
    b = int(((j2 == truth) & (tf != truth)).sum()); c = int(((tf == truth) & (j2 != truth)).sum())
    rng = np.random.default_rng(20261022); diffs = []
    for _ in range(500):
        s = rng.integers(0, len(ids), len(ids))
        diffs.append(max_share(truth[s], j2[s]) - max_share(truth[s], tf[s]))
    ref = np.bincount(truth, minlength=5) / len(truth)
    d1 = jev_labels(a.source / "jev-responses.jsonl")
    dev_ids = [i for i in d1 if by_id[i]["split"] == "dev"]
    p1 = acc_correct([IX[by_id[i]["label"]] for i in dev_ids], [d1[i] for i in dev_ids], j1)
    d2 = jev_labels(a.dev_v2 / "jev-responses.jsonl")
    p2 = acc_correct([IX[by_id[i]["label"]] for i in d2], [d2[i] for i in d2], j2)
    out = {
        "note": "Exploratory, not pre-registered. Pre-registered comparison is Jev v2 vs v1 (fresh-metrics.json).",
        "inputs_sha256": {"fresh_study": digest(a.fresh / "study.jsonl")},
        "jev_v2_vs_tfidf": {
            "accuracy": {"jev_v2": float((j2 == truth).mean()), "tfidf": float((tf == truth).mean())},
            "mcnemar": {"jev_v2_only_right": b, "tfidf_only_right": c, "exact_p": float(binomtest(b, b + c).pvalue)},
            "max_abs_share_error_pp": {"jev_v2": max_share(truth, j2), "tfidf": max_share(truth, tf),
                                        "paired_diff_ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])]},
        },
        "acc_correction_on_fresh": {
            "jev_v1_cc_max_pp": max_share(truth, j1),
            "jev_v1_acc_max_pp": float(100 * np.abs(p1 - ref).max()),
            "jev_v1_acc_signed_pp": dict(zip(LABELS, [float(x) for x in 100 * (p1 - ref)])),
            "jev_v2_cc_max_pp": max_share(truth, j2),
            "jev_v2_acc_max_pp": float(100 * np.abs(p2 - ref).max()),
            "confusion_source": {"jev_v1": "original dev split (v1 definitions)", "jev_v2": "results/dev-v2 (dev informed v2 design)"},
        },
    }
    (a.fresh / "fresh-extras.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
