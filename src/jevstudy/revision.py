#!/usr/bin/env python3
"""Pre-registered definitions revision: dev design, frozen fresh test, repeat check.

Every billable call goes through jev_runner.run (journal, budget, resume, lock).
Without --execute-jev the run command makes zero model calls. The build and eval
commands never call the network.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from . import jev_runner
from .metrics import scalar_metrics
from .study import (LABELS, PINNED_MODEL, PRICE_PER_MILLION, digest, keyword, object_hash,
                   read_jsonl, request_for, validate_answer, write_jsonl)

from .paths import DEFINITIONS_DIR, PREREGISTRATION, RESULTS_DIR
V1_PATH = DEFINITIONS_DIR / "v1.json"
V2_PATH = DEFINITIONS_DIR / "v2.json"
PREREG_PATH = PREREGISTRATION
FREEZE_BEGIN = "<!-- FROZEN:BEGIN -->"
FREEZE_END = "<!-- FROZEN:END -->"
BOOTSTRAP_REPEATS = 500
BOOTSTRAP_SEED = 20261022
CR = LABELS.index("credit_reporting")
DC = LABELS.index("debt_collection")
CC = LABELS.index("credit_card")
CR_TARGET_PP = 3.0          # decision rule: |credit_reporting signed share error| must be below this
ACCURACY_TOLERANCE = 0.01   # decision rule: accuracy may fall by at most this (absolute)
TOKENS_PER_CALL = 780       # planning estimate of input tokens per request
BOOT_METRICS = ("accuracy", "macro_f1", "max_abs_share_error_pp", "credit_reporting_share_error_pp")


def now():
    return datetime.now(timezone.utc).isoformat()


def estimated_list_price_usd(calls):
    return calls * TOKENS_PER_CALL * PRICE_PER_MILLION / 1_000_000


def load_definitions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(data) != {"instructions", "criteria"}:
        raise ValueError("Definitions need exactly 'instructions' and 'criteria'")
    if set(data["criteria"]) != set(LABELS):
        raise ValueError("Definitions must cover exactly the five labels")
    if not str(data["instructions"]).strip() or not all(str(v).strip() for v in data["criteria"].values()):
        raise ValueError("Empty definition text")
    return data


def request_with(row, definitions=None):
    """v1 request when definitions is None (identical to study.request_for); otherwise swap the text."""
    request = request_for(row)
    if definitions is not None:
        question = request["questions"]["product"]
        question["instructions"] = definitions["instructions"]
        question["criteria"] = {label: definitions["criteria"][label] for label in LABELS}
    return request


def verify_source(source):
    source = Path(source)
    manifest = json.loads((source / "data-manifest.json").read_text())
    if digest(source / "study.jsonl") != manifest["study_sha256"]:
        raise ValueError(f"{source}: study.jsonl does not match its manifest")
    return manifest


def load_ok_responses(folder):
    path = Path(folder) / "jev-responses.jsonl"
    responses = {}
    if path.exists():
        for record in read_jsonl(path):
            if record["status"] == "ok":
                responses[record["id"]] = record  # jev_runner forbids duplicate ok IDs
    return responses


def arm_predictions(folder, rows):
    responses = load_ok_responses(folder)
    missing = [r["id"] for r in rows if r["id"] not in responses]
    if missing:
        raise ValueError(f"{folder}: {len(missing)} rows lack an ok Jev response")
    labels, probs = [], []
    for r in rows:
        answer = validate_answer(responses[r["id"]]["response"])
        labels.append(LABELS.index(answer["label"]))
        probs.append(answer["probabilities"])
    return np.array(labels), np.array(probs)


def metric_row(m):
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "max_abs_share_error_pp": m["max_abs_share_error_pp"],
            "credit_reporting_share_error_pp": m["share_error_pp"][CR]}


def summarize(truth, pred, probs=None):
    m = scalar_metrics(truth, pred, probs)
    row = metric_row(m)
    row.update({"n": m["n"],
                "share_error_pp": dict(zip(LABELS, m["share_error_pp"])),
                "debt_collection_to_credit_reporting": int(m["confusion"][DC][CR]),
                "credit_card_to_credit_reporting": int(m["confusion"][CC][CR]),
                "confusion_rows_true_cols_pred": m["confusion"]})
    return row


# ---------- pre-registration freeze ----------

def read_freeze(prereg=PREREG_PATH):
    text = Path(prereg).read_text(encoding="utf-8")
    if text.count(FREEZE_BEGIN) != 1 or text.count(FREEZE_END) != 1:
        raise ValueError("PREREGISTRATION.md needs exactly one FROZEN block")
    block = text.split(FREEZE_BEGIN, 1)[1].split(FREEZE_END, 1)[0]
    values = dict(re.findall(r"^([a-z0-9_]+): (\S.*?)\s*$", block, re.M))
    return values or None


def verify_frozen(v2_path=V2_PATH, prereg=PREREG_PATH):
    state = read_freeze(prereg)
    if not state or "definitions_sha256" not in state:
        raise ValueError("Refused: v2 is not frozen. Run `python -m jevstudy.revision freeze` first.")
    if digest(v2_path) != state["definitions_sha256"]:
        raise ValueError("Refused: definitions/v2.json changed after freezing.")
    return state


def freeze(definitions_path=V2_PATH, prereg=PREREG_PATH):
    definitions_path = Path(definitions_path)
    load_definitions(definitions_path)
    value = digest(definitions_path)
    state = read_freeze(prereg)
    if state and "definitions_sha256" in state:
        if state["definitions_sha256"] == value:
            return state
        raise ValueError("Already frozen with a different v2 hash; refusing to re-freeze.")
    text = Path(prereg).read_text(encoding="utf-8")
    before, rest = text.split(FREEZE_BEGIN, 1)
    _, after = rest.split(FREEZE_END, 1)
    new_state = {"definitions_file": "definitions/v2.json", "definitions_sha256": value, "frozen_utc": now()}
    block = FREEZE_BEGIN + "\n" + "\n".join(f"{k}: {v}" for k, v in new_state.items()) + "\n" + FREEZE_END
    Path(prereg).write_text(before + block + after, encoding="utf-8")
    copy = definitions_path.parent / "v2.frozen.json"
    if not copy.exists():
        copy.write_bytes(definitions_path.read_bytes())
    return new_state


# ---------- dev-only v2 requests ----------

def build_dev_v2(source, out, v2_path=V2_PATH, dry_run=False):
    source, out = Path(source), Path(out)
    verify_source(source)
    definitions = load_definitions(v2_path)
    rows = [r for r in read_jsonl(source / "study.jsonl") if r["split"] == "dev"]
    requests = [{"id": r["id"], "split": "dev", "request": request_with(r, definitions)} for r in rows]
    summary = {"mode": "dry_run" if dry_run else "build", "requests": len(requests), "model_calls": 0,
               "estimated_list_price_usd": estimated_list_price_usd(len(requests)),
               "definitions_sha256": digest(v2_path), "output": str(out)}
    if dry_run:
        print(json.dumps(summary, indent=2))
        return summary
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "study.jsonl", rows)
    write_jsonl(out / "requests.jsonl", requests)
    manifest = {"arm": "dev-v2", "created_utc": now(), "preregistered": False, "model": PINNED_MODEL,
                "source_study_sha256": digest(source / "study.jsonl"),
                "study_sha256": digest(out / "study.jsonl"), "requests_sha256": digest(out / "requests.jsonl"),
                "definitions_file": "definitions/v2.json", "definitions_sha256": summary["definitions_sha256"],
                "n": len(rows), "splits": {"dev": {"n": len(rows), "counts": dict(Counter(r["label"] for r in rows))}},
                "note": "Development-only requests. v2 was designed on these records; not evidence of improvement."}
    (out / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def run_arm(folder, budget=1.0, execute=False, max_requests=None, pause=.05, timeout=30.,
            retry_failed=False, fetcher=jev_runner.fetch, v2_path=V2_PATH, prereg=PREREG_PATH):
    folder = Path(folder)
    manifest = json.loads((folder / "data-manifest.json").read_text())
    if manifest.get("preregistered"):
        state = verify_frozen(v2_path, prereg)
        if manifest.get("arm") == "v2" and manifest.get("definitions_sha256") != state["definitions_sha256"]:
            raise ValueError("Arm manifest does not match the frozen v2 definitions")
        if manifest.get("arm") == "v1" and manifest.get("definitions_sha256") != digest(V1_PATH):
            raise ValueError("Arm manifest does not match definitions/v1.json")
    return jev_runner.run(folder, budget=budget, execute=execute, max_requests=max_requests, pause=pause,
                          timeout=timeout, retry_failed=retry_failed, fetcher=fetcher)


def eval_dev(v1_folder, v2_folder):
    v1_folder, v2_folder = Path(v1_folder), Path(v2_folder)
    v1_rows = [r for r in read_jsonl(v1_folder / "study.jsonl") if r["split"] == "dev"]
    verify_source(v2_folder)
    v2_rows = read_jsonl(v2_folder / "study.jsonl")
    if {r["id"] for r in v1_rows} != {r["id"] for r in v2_rows}:
        raise ValueError("Dev IDs differ between v1 and v2 folders")
    truth = np.array([LABELS.index(r["label"]) for r in v1_rows])
    p1, q1 = arm_predictions(v1_folder, v1_rows)
    p2, q2 = arm_predictions(v2_folder, v1_rows)
    s1, s2 = summarize(truth, p1, q1), summarize(truth, p2, q2)
    keys = ("accuracy", "macro_f1", "max_abs_share_error_pp", "credit_reporting_share_error_pp",
            "debt_collection_to_credit_reporting")
    result = {"split": "dev", "n": len(v1_rows),
              "caveat": "Dev records informed the v2 design. This is not a confirmatory result.",
              "v1": {k: s1[k] for k in keys + ("share_error_pp",)},
              "v2": {k: s2[k] for k in keys + ("share_error_pp",)},
              "v2_minus_v1": {k: s2[k] - s1[k] for k in keys}}
    (v2_folder / "dev-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ---------- fresh set ----------

def build_fresh(fresh_root, v2_path=V2_PATH, prereg=PREREG_PATH):
    root = Path(fresh_root)
    state = verify_frozen(v2_path, prereg)
    manifest = json.loads((root / "data-manifest.json").read_text())
    if not manifest.get("fresh"):
        raise ValueError("Not a fresh study folder")
    if digest(root / "study.jsonl") != manifest["study_sha256"]:
        raise ValueError("Fresh study.jsonl does not match its manifest")
    if digest(root / "requests.jsonl") != manifest["requests_sha256"]:
        raise ValueError("Fresh requests.jsonl does not match its manifest")
    rows = read_jsonl(root / "study.jsonl")
    v2 = load_definitions(v2_path)
    v1_requests = [{"id": r["id"], "split": "fresh", "request": request_with(r)} for r in rows]
    if v1_requests != read_jsonl(root / "requests.jsonl"):
        raise ValueError("Rebuilt v1 requests differ from the prepared requests")
    v2_requests = [{"id": r["id"], "split": "fresh", "request": request_with(r, v2)} for r in rows]
    summaries = {}
    for arm, requests, defs_sha, defs_file in (
            ("v1", v1_requests, digest(V1_PATH), "definitions/v1.json"),
            ("v2", v2_requests, state["definitions_sha256"], "definitions/v2.json")):
        folder = root / f"jev-{arm}"
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"{folder} is not empty")
        folder.mkdir(parents=True, exist_ok=True)
        write_jsonl(folder / "requests.jsonl", requests)
        arm_manifest = {"arm": arm, "preregistered": True, "created_utc": now(), "model": PINNED_MODEL,
                        "n": len(requests), "study_sha256": manifest["study_sha256"],
                        "requests_sha256": digest(folder / "requests.jsonl"),
                        "definitions_file": defs_file, "definitions_sha256": defs_sha,
                        "parent_manifest_sha256": digest(root / "data-manifest.json"),
                        "estimated_list_price_usd": estimated_list_price_usd(len(requests))}
        (folder / "data-manifest.json").write_text(json.dumps(arm_manifest, indent=2) + "\n")
        summaries[arm] = {"folder": str(folder), "requests": len(requests),
                          "requests_sha256": arm_manifest["requests_sha256"]}
    return summaries


def fit_tfidf(train_rows):
    # Identical to baselines.py: same vectoriser, classifier, limits and warning policy.
    pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)),
                         ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1500, random_state=20261022))])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit([r["text"] for r in train_rows], [r["label"] for r in train_rows])
    indices = [list(pipeline.classes_).index(label) for label in LABELS]
    return pipeline, indices


def predict_tfidf(pipeline, indices, rows):
    labels, probs = [], []
    with threadpool_limits(limits=1):
        for r in rows:
            p = pipeline.predict_proba([r["text"]])[0][indices].tolist()
            probs.append(p)
            labels.append(int(np.argmax(p)))
    return np.array(labels), np.array(probs)


def check_tfidf_reproduces(source, pipeline, indices):
    # The refit must reproduce the original baseline predictions before it is used on fresh data.
    stored = {(r["id"], r["split"]): r["label"] for r in read_jsonl(source / "local-predictions.jsonl")
              if r["method"] == "tfidf"}
    rows = [r for r in read_jsonl(source / "study.jsonl") if r["split"] != "train"]
    labels, _ = predict_tfidf(pipeline, indices, rows)
    matches = sum(LABELS[lab] == stored[(r["id"], r["split"])] for r, lab in zip(rows, labels))
    if matches != len(rows):
        raise ValueError(f"Refit tf-idf reproduces only {matches}/{len(rows)} original predictions; refusing")
    return {"checked": len(rows), "matches": matches}


def bootstrap_arms(truth, arms, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    # One resample index per repeat, shared by every arm, so differences are paired.
    rng = np.random.default_rng(seed)
    values = {name: {k: [] for k in BOOT_METRICS} for name in arms}
    n = len(truth)
    for _ in range(repeats):
        ix = rng.integers(0, n, n)
        for name, (pred, _probs) in arms.items():
            row = metric_row(scalar_metrics(truth[ix], pred[ix]))
            for k in BOOT_METRICS:
                values[name][k].append(row[k])
    return values


def eval_fresh(fresh_root, train_source, v2_path=V2_PATH, prereg=PREREG_PATH):
    root, train_source = Path(fresh_root), Path(train_source)
    state = verify_frozen(v2_path, prereg)
    manifest = json.loads((root / "data-manifest.json").read_text())
    if not manifest.get("fresh") or digest(root / "study.jsonl") != manifest["study_sha256"]:
        raise ValueError("Fresh study folder failed its manifest check")
    for arm in ("v1", "v2"):
        am = json.loads((root / f"jev-{arm}" / "data-manifest.json").read_text())
        if (am.get("preregistered") is not True or am["study_sha256"] != manifest["study_sha256"]
                or am["requests_sha256"] != digest(root / f"jev-{arm}" / "requests.jsonl")):
            raise ValueError(f"jev-{arm} manifest does not match the fresh study")
        if arm == "v2" and am["definitions_sha256"] != state["definitions_sha256"]:
            raise ValueError("jev-v2 was built from definitions other than the frozen v2")
    rows = read_jsonl(root / "study.jsonl")
    truth = np.array([LABELS.index(r["label"]) for r in rows])
    verify_source(train_source)
    train = [r for r in read_jsonl(train_source / "study.jsonl") if r["split"] == "train"]
    pipeline, indices = fit_tfidf(train)
    reproduced = check_tfidf_reproduces(train_source, pipeline, indices)
    preds = {
        "keyword": (np.array([LABELS.index(keyword(r["text"])) for r in rows]), None),
        "tfidf": predict_tfidf(pipeline, indices, rows),
        "jev_v1": arm_predictions(root / "jev-v1", rows),
        "jev_v2": arm_predictions(root / "jev-v2", rows),
    }
    point = {name: summarize(truth, pred) for name, (pred, _p) in preds.items()}
    boot = bootstrap_arms(truth, preds)
    ci = {name: {k: np.quantile(boot[name][k], [.025, .975]).tolist() for k in BOOT_METRICS} for name in preds}
    diff = {k: np.array(boot["jev_v2"][k]) - np.array(boot["jev_v1"][k]) for k in BOOT_METRICS}
    paired = {"point": {k: point["jev_v2"][k] - point["jev_v1"][k] for k in BOOT_METRICS},
              "ci95": {k: np.quantile(diff[k], [.025, .975]).tolist() for k in BOOT_METRICS}}
    cr_v2 = point["jev_v2"]["credit_reporting_share_error_pp"]
    acc_change = paired["point"]["accuracy"]
    decision = {"credit_reporting_within_target": bool(abs(cr_v2) < CR_TARGET_PP),
                "accuracy_change": acc_change,
                "accuracy_within_tolerance": bool(acc_change >= -ACCURACY_TOLERANCE),
                "fixes_bias_claim": bool(abs(cr_v2) < CR_TARGET_PP and acc_change >= -ACCURACY_TOLERANCE)}
    result = {"fresh_window": manifest["fresh_window"], "window": [manifest["window_start"], manifest["window_end_exclusive"]],
              "n": len(rows), "study_sha256": manifest["study_sha256"],
              "frozen_definitions_sha256": state["definitions_sha256"],
              "train_source": str(train_source), "tfidf_reproduction": reproduced,
              "arms": point, "bootstrap_ci95": ci,
              "paired_v2_minus_v1": paired,
              "primary": {"max_abs_share_error_pp": {"v1": point["jev_v1"]["max_abs_share_error_pp"],
                                                     "v2": point["jev_v2"]["max_abs_share_error_pp"],
                                                     "v2_minus_v1": paired["point"]["max_abs_share_error_pp"],
                                                     "ci95_diff": paired["ci95"]["max_abs_share_error_pp"]},
                          "credit_reporting_signed_share_error_pp": {"v1": point["jev_v1"]["credit_reporting_share_error_pp"],
                                                                     "v2": cr_v2}},
              "decision_rule": (f"fixes_bias_claim = |v2 credit_reporting signed share error| < {CR_TARGET_PP} pp "
                                f"AND v2 accuracy - v1 accuracy >= -{ACCURACY_TOLERANCE}"),
              "decision": decision,
              "bootstrap": {"repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
                            "method": "nonparametric resample of fresh records, shared across arms"}}
    (root / "fresh-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ---------- repeat check ----------

def build_repeat(source, out, n=200, copies=2, seed_tag="repeat-20261023", dry_run=False):
    source, out = Path(source), Path(out)
    verify_source(source)
    original_requests = {r["id"]: r for r in read_jsonl(source / "requests.jsonl")}
    test_rows = [r for r in read_jsonl(source / "study.jsonl") if r["split"] == "test"]
    chosen = sorted(test_rows, key=lambda r: object_hash([seed_tag, r["id"]]))[:n]
    chosen.sort(key=lambda r: (r["date"], int(r["id"])))
    rows, requests = [], []
    for r in chosen:
        original = original_requests[r["id"]]
        if original["request"] != request_with(r):
            raise ValueError("Stored v1 request differs from the rebuilt v1 request")
        for k in range(1, copies + 1):
            new_id = f"{r['id']}#r{k}"
            rows.append({**r, "id": new_id, "split": "repeat", "original_id": r["id"], "copy": k})
            requests.append({"id": new_id, "split": "repeat", "request": original["request"]})
    summary = {"mode": "dry_run" if dry_run else "build", "requests": len(requests), "model_calls": 0,
               "estimated_list_price_usd": estimated_list_price_usd(len(requests)), "output": str(out)}
    if dry_run:
        print(json.dumps(summary, indent=2))
        return summary
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "study.jsonl", rows)
    write_jsonl(out / "requests.jsonl", requests)
    manifest = {"arm": "repeat", "created_utc": now(), "preregistered": False, "model": PINNED_MODEL,
                "source_study_sha256": digest(source / "study.jsonl"),
                "study_sha256": digest(out / "study.jsonl"), "requests_sha256": digest(out / "requests.jsonl"),
                "n": len(requests), "originals": len(chosen), "copies": copies,
                "original_ids_sha256": object_hash(sorted(r["id"] for r in chosen)),
                "note": "Requests are byte-identical to the v1 originals. Measures run-to-run stability only."}
    (out / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def eval_repeat(repeat_folder, source):
    repeat, source = Path(repeat_folder), Path(source)
    manifest = json.loads((repeat / "data-manifest.json").read_text())
    if digest(repeat / "study.jsonl") != manifest["study_sha256"]:
        raise ValueError("Repeat study.jsonl does not match its manifest")
    rows = read_jsonl(repeat / "study.jsonl")
    originals = load_ok_responses(source)
    copies = load_ok_responses(repeat)
    missing = [r["id"] for r in rows if r["id"] not in copies or r["original_id"] not in originals]
    if missing:
        raise ValueError(f"{len(missing)} repeat rows lack an ok response on one side")
    agree, diffs = [], []
    for r in rows:
        a = validate_answer(originals[r["original_id"]]["response"])
        b = validate_answer(copies[r["id"]]["response"])
        agree.append(a["label"] == b["label"])
        diffs.append(np.abs(np.array(a["probabilities"]) - np.array(b["probabilities"])))
    D = np.array(diffs)
    agree = np.array(agree)
    result = {"repeat_responses": len(rows), "originals": manifest["originals"], "copies": manifest["copies"],
              "label_agreement": float(agree.mean()), "label_disagreements": int((~agree).sum()),
              "mean_abs_probability_difference": float(D.mean()),
              "max_abs_probability_difference": float(D.max()), "model_calls": 0}
    (repeat / "repeat-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-dev-v2", help="Dev-only v2 requests (no model calls)")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("run", help="Run one prepared folder; zero calls unless --execute-jev")
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--execute-jev", action="store_true")
    p.add_argument("--budget-usd", type=float, default=1.0)
    p.add_argument("--max-requests", type=int)
    p.add_argument("--pause", type=float, default=.05)
    p.add_argument("--timeout", type=float, default=30.)
    p.add_argument("--retry-failed", action="store_true")

    p = sub.add_parser("eval-dev", help="v1 vs v2 on dev")
    p.add_argument("--v1", type=Path, required=True)
    p.add_argument("--v2", type=Path, required=True)

    p = sub.add_parser("freeze", help="Freeze definitions/v2.json in PREREGISTRATION.md")
    p.add_argument("--definitions", type=Path, default=V2_PATH)

    p = sub.add_parser("build-fresh", help="Requests for v1 and v2 on the fresh set (requires freeze)")
    p.add_argument("--fresh", type=Path, required=True)

    p = sub.add_parser("eval-fresh", help="Score keyword, tf-idf and Jev v1/v2 on the fresh set")
    p.add_argument("--fresh", type=Path, required=True)
    p.add_argument("--train-source", type=Path, default=RESULTS_DIR / "main")

    p = sub.add_parser("build-repeat", help="Repeat-stability requests (no model calls)")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("eval-repeat", help="Agreement of repeat responses with the originals")
    p.add_argument("--repeat", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "build-dev-v2":
        return build_dev_v2(args.source, args.out, dry_run=args.dry_run)
    if args.command == "run":
        if args.max_requests is not None and args.max_requests < 1:
            parser.error("--max-requests must be positive")
        outcome = run_arm(args.folder, budget=args.budget_usd, execute=args.execute_jev,
                          max_requests=args.max_requests, pause=args.pause, timeout=args.timeout,
                          retry_failed=args.retry_failed)
        if args.execute_jev and outcome["status"] != "complete":
            raise SystemExit(2)
        return outcome
    if args.command == "eval-dev":
        result = eval_dev(args.v1, args.v2)
        print(json.dumps(result, indent=2))
        return result
    if args.command == "freeze":
        if args.definitions.resolve() != V2_PATH.resolve():
            parser.error("Freeze applies only to definitions/v2.json")
        state = freeze(args.definitions)
        print(json.dumps(state, indent=2))
        return state
    if args.command == "build-fresh":
        summaries = build_fresh(args.fresh)
        print(json.dumps(summaries, indent=2))
        return summaries
    if args.command == "eval-fresh":
        result = eval_fresh(args.fresh, args.train_source)
        print(json.dumps({"decision": result["decision"], "primary": result["primary"]}, indent=2))
        return result
    if args.command == "build-repeat":
        if args.n < 1 or args.copies < 1:
            parser.error("--n and --copies must be positive")
        return build_repeat(args.source, args.out, args.n, args.copies, dry_run=args.dry_run)
    if args.command == "eval-repeat":
        result = eval_repeat(args.repeat, args.source)
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    main()
