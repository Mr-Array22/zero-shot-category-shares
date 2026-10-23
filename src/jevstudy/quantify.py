#!/usr/bin/env python3
"""Standard quantification corrections for category-share dashboards.

Corrections are estimated only from development data (labels and, where
available, probabilities) and applied to held-out test predictions. Test
reference labels are used only to score corrected shares, never to fit them.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from .study import LABELS, NAMES, digest, object_hash, read_jsonl, validate_answer

K = len(LABELS)
SEED = 20261022
REPEATS = 500
SIMPLEX_WEIGHT = 1e6
CLIP_TOL = 1e-12
EM_TOL = 1e-10
EM_MAX_ITER = 1000
MODELS = ("keyword", "tfidf", "jev")
METHODS = ("CC", "ACC_clip", "ACC_simplex", "PCC", "PACC_clip", "PACC_simplex", "EM")
DEFINITIONS = {
    "CC": "Classify and count: share of test predictions per category (hard labels).",
    "ACC_clip": "Adjusted classify and count (Forman): solve M p = q with M estimated on dev; clip negative components to zero and renormalise.",
    "ACC_simplex": "ACC with the solution constrained to the probability simplex: weighted NNLS with a sum-to-one row, then renormalise.",
    "PCC": "Probabilistic classify and count: mean predicted probability vector on test.",
    "PACC_clip": "Probabilistic ACC (Bella et al.): M from mean dev probability vectors per true class; clip and renormalise.",
    "PACC_simplex": "PACC with simplex-constrained least squares (weighted NNLS, then renormalise).",
    "EM": "Saerens, Latinne and Decaestecker (2002) EM prior adjustment; training priors = dev class proportions; assumes calibrated probabilities.",
}


def select_jev_responses(attempts, requests, expected):
    """Keep the last successful response per id; failed attempts are ignored."""
    selected = {}
    ok_counts = Counter()
    for record in attempts:
        if record["status"] != "ok":
            continue
        rid = record["id"]
        if rid not in expected or rid not in requests or record["request_hash"] != object_hash(requests[rid]["request"]):
            raise ValueError(f"Mismatched Jev response for id {rid}")
        answer = validate_answer(record["response"])
        ok_counts[rid] += 1
        selected[rid] = answer
    meta = {"failed_attempt_records": int(sum(r["status"] != "ok" for r in attempts)),
            "ids_with_multiple_successes": int(sum(v > 1 for v in ok_counts.values())),
            "successful_records_used": int(len(selected))}
    return selected, meta


def split_view(rows, predictions):
    y = np.array([LABELS.index(r["label"]) for r in rows], dtype=int)
    pred = np.array([LABELS.index(predictions[r["id"]]["label"]) for r in rows], dtype=int)
    raw = [predictions[r["id"]].get("probabilities") for r in rows]
    P = None
    if all(p is not None for p in raw):
        P = np.array(raw, dtype=float)
        if P.shape != (len(rows), K) or not np.isfinite(P).all() or (P < 0).any() or not np.allclose(P.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Invalid probability matrix")
    return {"y": y, "pred": pred, "P": P}


def load_run(folder):
    folder = Path(folder)
    data = json.loads((folder / "data-manifest.json").read_text())
    if digest(folder / "study.jsonl") != data["study_sha256"] or digest(folder / "requests.jsonl") != data["requests_sha256"]:
        raise ValueError("Frozen study/request hash mismatch")
    rows = read_jsonl(folder / "study.jsonl")
    by_id = {r["id"]: r for r in rows}
    dev_rows = [r for r in rows if r["split"] == "dev"]
    test_rows = [r for r in rows if r["split"] == "test"]
    expected = {r["id"] for r in dev_rows + test_rows}
    local = json.loads((folder / "local-manifest.json").read_text())
    if digest(folder / "local-predictions.jsonl") != local["predictions_sha256"] or local["study_sha256"] != data["study_sha256"]:
        raise ValueError("Local predictions do not match frozen study")
    local_preds = {"keyword": {}, "tfidf": {}}
    for record in read_jsonl(folder / "local-predictions.jsonl"):
        method, rid = record["method"], record["id"]
        if method not in local_preds or rid in local_preds[method] or rid not in expected or record["input_hash"] != object_hash(by_id[rid]["text"]):
            raise ValueError("Invalid or duplicate local prediction")
        local_preds[method][rid] = record
    for method in local_preds:
        if set(local_preds[method]) != expected:
            raise ValueError(f"Incomplete local predictions for {method}")
    requests = {r["id"]: r for r in read_jsonl(folder / "requests.jsonl")}
    attempts = read_jsonl(folder / "jev-responses.jsonl")
    jev_answers, jev_meta = select_jev_responses(attempts, requests, expected)
    if set(jev_answers) != expected:
        raise ValueError(f"Jev results incomplete: {len(jev_answers)} of {len(expected)}")
    predictions = {"keyword": local_preds["keyword"], "tfidf": local_preds["tfidf"],
                   "jev": {rid: {"label": a["label"], "probabilities": a["probabilities"]} for rid, a in jev_answers.items()}}
    views = {m: {"dev": split_view(dev_rows, predictions[m]), "test": split_view(test_rows, predictions[m])} for m in MODELS}
    for m in MODELS:
        if not np.array_equal(views[m]["test"]["y"], views["tfidf"]["test"]["y"]):
            raise ValueError("Reference labels differ between models")
    evidence = {name: digest(folder / name) for name in ("study.jsonl", "requests.jsonl", "local-predictions.jsonl",
                                                          "jev-responses.jsonl", "metrics.json") if (folder / name).exists()}
    return {"views": views, "data": data, "jev_meta": jev_meta, "evidence": evidence}


def class_counts(y, k=K):
    return np.bincount(np.asarray(y, int), minlength=k)


def truth_share(y, k=K):
    return class_counts(y, k) / len(y)


def hard_confusion(y, pred, k=K):
    """M[j, i] = P(predicted j | true i), estimated from labelled data."""
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    counts = class_counts(y, k)
    if (counts == 0).any():
        raise ValueError(f"Development split lacks a category: {counts.tolist()}")
    joint = np.bincount(pred * k + y, minlength=k * k).reshape(k, k).astype(float)
    return joint / counts


def soft_confusion(y, P):
    """M[:, i] = mean probability vector over development items whose true class is i."""
    y = np.asarray(y, int)
    k = P.shape[1]
    counts = class_counts(y, k)
    if (counts == 0).any():
        raise ValueError(f"Development split lacks a category: {counts.tolist()}")
    return np.column_stack([P[y == i].mean(axis=0) for i in range(k)])


def solve_clip(M, q):
    raw = np.linalg.solve(M, q)
    clipped = np.clip(raw, 0, None)
    estimate = clipped / clipped.sum()
    diag = {"clip_needed": bool(raw.min() < -CLIP_TOL), "min_raw": float(raw.min()),
            "raw_solution": raw.tolist(), "residual_norm": float(np.linalg.norm(M @ estimate - q))}
    return estimate, diag


def solve_simplex(M, q):
    k = M.shape[1]
    A = np.vstack([M, SIMPLEX_WEIGHT * np.ones((1, k))])
    b = np.concatenate([q, [SIMPLEX_WEIGHT]])
    raw, _ = nnls(A, b)
    total = float(raw.sum())
    estimate = raw / total
    diag = {"sum_before_renormalise": total, "residual_norm": float(np.linalg.norm(M @ estimate - q))}
    return estimate, diag


def em_prior(P, train_prior, tol=EM_TOL, max_iter=EM_MAX_ITER):
    train_prior = np.asarray(train_prior, float)
    if (train_prior <= 0).any():
        raise ValueError("Training priors must be positive")
    prior = train_prior.copy()
    change = float("nan")
    for iteration in range(1, max_iter + 1):
        weights = np.maximum(prior, 1e-12) / train_prior
        posterior = P * weights
        posterior /= posterior.sum(axis=1, keepdims=True)
        updated = posterior.mean(axis=0)
        change = float(np.abs(updated - prior).max())
        prior = updated
        if change < tol:
            return prior, {"iterations": iteration, "converged": True, "final_change": change}
    return prior, {"iterations": max_iter, "converged": False, "final_change": change}


def estimate_all(dev, test):
    """Fit every correction on dev and apply it to test. Returns (estimates, diagnostics)."""
    M = hard_confusion(dev["y"], dev["pred"])
    q = class_counts(test["pred"], M.shape[0]) / len(test["pred"])
    est = {"CC": q}
    info = {"methods": {}}
    est["ACC_clip"], info["methods"]["ACC_clip"] = solve_clip(M, q)
    est["ACC_simplex"], info["methods"]["ACC_simplex"] = solve_simplex(M, q)
    info["cond_hard"] = float(np.linalg.cond(M))
    if dev["P"] is not None and test["P"] is not None:
        qs = test["P"].mean(axis=0)
        Ms = soft_confusion(dev["y"], dev["P"])
        est["PCC"] = qs
        est["PACC_clip"], info["methods"]["PACC_clip"] = solve_clip(Ms, qs)
        est["PACC_simplex"], info["methods"]["PACC_simplex"] = solve_simplex(Ms, qs)
        info["cond_soft"] = float(np.linalg.cond(Ms))
        prior = class_counts(dev["y"], M.shape[0]) / len(dev["y"])
        est["EM"], info["methods"]["EM"] = em_prior(test["P"], prior)
    return est, info


def errors(estimate, truth):
    delta = np.asarray(estimate, float) - truth
    return {"signed_share_error_pp": (100 * delta).tolist(),
            "max_abs_share_error_pp": float(100 * np.abs(delta).max()),
            "total_variation": float(0.5 * np.abs(delta).sum())}


def interval(values):
    if len(values) == 0:
        return [None, None]
    low, high = np.quantile(np.asarray(values, float), [.025, .975])
    return [float(low), float(high)]


def subset(view, ix):
    return {"y": view["y"][ix], "pred": view["pred"][ix], "P": None if view["P"] is None else view["P"][ix]}


def bootstrap(views, repeats=REPEATS, seed=SEED):
    """Resample dev and test independently; identical indices drive every model (paired)."""
    rng = np.random.default_rng(seed)
    n_dev = len(views["tfidf"]["dev"]["y"])
    n_test = len(views["tfidf"]["test"]["y"])
    maxabs = {m: {} for m in MODELS}
    clip = {m: Counter() for m in MODELS}
    nonconverged = Counter()
    valid = skipped_support = skipped_fit = 0
    for _ in range(repeats):
        ix_dev = rng.integers(0, n_dev, n_dev)
        ix_test = rng.integers(0, n_test, n_test)
        if (class_counts(views["tfidf"]["dev"]["y"][ix_dev]) == 0).any():
            skipped_support += 1
            continue
        try:
            fitted = {m: estimate_all(subset(views[m]["dev"], ix_dev), subset(views[m]["test"], ix_test)) for m in MODELS}
        except (np.linalg.LinAlgError, ValueError):
            skipped_fit += 1
            continue
        valid += 1
        truth = truth_share(views["tfidf"]["test"]["y"][ix_test])
        for m, (est, info) in fitted.items():
            for method, vec in est.items():
                maxabs[m].setdefault(method, []).append(float(100 * np.abs(vec - truth).max()))
                diag = info["methods"].get(method, {})
                if diag.get("clip_needed"):
                    clip[m][method] += 1
            if "EM" in info["methods"] and not info["methods"]["EM"]["converged"]:
                nonconverged[m] += 1
    return {"maxabs": maxabs, "clip": clip, "nonconverged": nonconverged, "valid": valid,
            "skipped_dev_missing_category": skipped_support, "skipped_fit_error": skipped_fit}


def diagnostics(views):
    dev_y = views["tfidf"]["dev"]["y"]
    test_y = views["tfidf"]["test"]["y"]
    dev_counts = class_counts(dev_y)
    test_counts = class_counts(test_y)
    dev_prior = dev_counts / dev_counts.sum()
    test_prior = test_counts / test_counts.sum()
    out = {"dev_class_counts": dict(zip(LABELS, dev_counts.tolist())),
           "test_class_counts": dict(zip(LABELS, test_counts.tolist())),
           "dev_priors": dict(zip(LABELS, dev_prior.tolist())),
           "test_priors": dict(zip(LABELS, test_prior.tolist())),
           "prior_shift_total_variation": float(0.5 * np.abs(dev_prior - test_prior).sum()),
           "smallest_dev_class_count": int(dev_counts.min()),
           "conditional_shift": {}, "probability_structure": {}}
    for m in MODELS:
        dev, test = views[m]["dev"], views[m]["test"]
        M_dev = hard_confusion(dev["y"], dev["pred"])
        M_test = hard_confusion(test["y"], test["pred"])
        out["conditional_shift"][m] = {
            "max_abs_entry_difference_dev_vs_test": float(np.abs(M_dev - M_test).max()),
            "note": "Diagnostic only. Uses test labels to judge whether P(prediction | category) is stable; never used for a correction."}
        out["probability_structure"][m] = {}
        for split, view in (("dev", dev), ("test", test)):
            if view["P"] is not None:
                out["probability_structure"][m][split] = {
                    "share_top_probability_at_least_0.995": float((view["P"].max(axis=1) >= .995).mean()),
                    "share_entries_exactly_zero": float((view["P"] == 0).mean())}
    return out


def analyse(loaded, repeats=REPEATS):
    views = loaded["views"]
    truth = truth_share(views["tfidf"]["test"]["y"])
    dev_truth = truth_share(views["tfidf"]["dev"]["y"])
    point = {m: estimate_all(views[m]["dev"], views[m]["test"]) for m in MODELS}
    boot = bootstrap(views, repeats)
    models = {}
    for m in MODELS:
        est, info = point[m]
        dev_cc = errors(class_counts(views[m]["dev"]["pred"]) / len(views[m]["dev"]["pred"]), dev_truth)
        entry = {"available_methods": list(est), "condition_number_hard": info["cond_hard"],
                 "condition_number_soft": info.get("cond_soft"),
                 "dev_cc_signed_share_error_pp": dev_cc["signed_share_error_pp"],
                 "dev_cc_max_abs_share_error_pp": dev_cc["max_abs_share_error_pp"], "methods": {}}
        for method, vec in est.items():
            row = {"estimate_share": vec.tolist(), **errors(vec, truth),
                   "max_abs_share_error_pp_ci95": interval(boot["maxabs"][m][method]),
                   "bootstrap_valid_replicates": len(boot["maxabs"][m][method])}
            if method in info["methods"]:
                row["diagnostics"] = info["methods"][method]
            if method in ("ACC_clip", "PACC_clip"):
                row["bootstrap_clip_needed_fraction"] = clip_fraction(boot, m, method)
            entry["methods"][method] = row
        models[m] = entry
    paired = {}
    for method in METHODS:
        if method in models["jev"]["methods"] and method in models["tfidf"]["methods"]:
            diff = models["jev"]["methods"][method]["max_abs_share_error_pp"] - models["tfidf"]["methods"][method]["max_abs_share_error_pp"]
            draws = np.array(boot["maxabs"]["jev"][method]) - np.array(boot["maxabs"]["tfidf"][method])
            paired[method] = {"point_jev_minus_tfidf_pp": float(diff), "ci95_jev_minus_tfidf_pp": interval(draws)}
    return {"models": models, "paired_jev_minus_tfidf": paired,
            "bootstrap": {"unit": "individual deduplicated narrative; dev and test resampled independently with replacement",
                          "repeats": repeats, "seed": SEED, "valid_replicates": boot["valid"],
                          "skipped_dev_missing_category": boot["skipped_dev_missing_category"],
                          "skipped_fit_error": boot["skipped_fit_error"],
                          "em_nonconverged_replicates": dict(boot["nonconverged"]),
                          "ci": "percentile 2.5 and 97.5"},
            "truth_test_share": dict(zip(LABELS, truth.tolist())),
            "truth_dev_share": dict(zip(LABELS, dev_truth.tolist()))}


def clip_fraction(boot, m, method):
    valid = max(len(boot["maxabs"][m].get(method, [])), 1)
    return boot["clip"][m][method] / valid


def check_against_metrics(folder, models):
    path = Path(folder) / "metrics.json"
    if not path.exists():
        return {"checked": False}
    metrics = json.loads(path.read_text())["methods"]
    checked = []
    for m in MODELS:
        if m not in metrics:
            continue
        reference = np.array(metrics[m]["share_error_pp"], float)
        ours = np.array(models[m]["methods"]["CC"]["signed_share_error_pp"], float)
        if not np.allclose(reference, ours, atol=1e-9, rtol=0):
            raise ValueError(f"CC share errors for {m} do not reproduce metrics.json")
        checked.append(m)
    return {"checked": True, "models_reproduced": checked, "reference": "metrics.json share_error_pp"}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def draw(folder, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.text
    from matplotlib.image import imread
    from .figures import COLORS, NICK
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white",
                         "savefig.facecolor": "white", "svg.fonttype": "none"})
    out = Path(folder) / "figures"
    out.mkdir(exist_ok=True)
    groups = [("CC", "CC"), ("ACC_clip", "ACC\n(clip)"), ("ACC_simplex", "ACC\n(simplex)"), ("PCC", "PCC"),
              ("PACC_clip", "PACC\n(clip)"), ("PACC_simplex", "PACC\n(simplex)"), ("EM", "EM prior\nadjustment")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(groups))
    width = .26
    tops = []
    bars = []
    for k, m in enumerate(MODELS):
        for g, (method, _) in enumerate(groups):
            row = result["models"][m]["methods"].get(method)
            xpos = x[g] + (k - 1) * width
            if row is None:
                ax.text(xpos, .05, "n/a", rotation=90, ha="center", va="bottom", fontsize=8, color="#6B7280")
                continue
            value = row["max_abs_share_error_pp"]
            low, high = row["max_abs_share_error_pp_ci95"]
            bars.append((xpos, value, high))
            ax.bar(xpos, value, width=width * .9, color=COLORS[m], label=NICK[m] if g == 0 else None)
            ax.errorbar(xpos, value, yerr=[[value - low], [high - value]], fmt="none", ecolor="#1F2937", capsize=2.5, lw=1)
            tops.append(high)
    ymax = max(tops) * 1.4
    offset = .02 * ymax
    for xpos, value, high in bars:
        ax.text(xpos, high + offset, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x, [label for _, label in groups], fontsize=9)
    ax.set_ylabel("Largest absolute share error on test (pp)")
    ax.set_title("Do standard quantification corrections bring dashboard shares closer to test references?", fontsize=12)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0), fontsize=9, frameon=False)
    footnote = ("Corrections estimated on development data only. Whiskers: 95% percentile bootstrap intervals from "
                f"{result['bootstrap']['repeats']} independent development and test resamples (seed {SEED}). "
                "Keyword rules output no probabilities, so only CC and ACC appear; n/a marks methods that do not apply.")
    wrapped = "\n".join(textwrap.wrap(footnote, width=int(fig.get_figwidth() * 14)))
    text = fig.text(.03, .015, wrapped, fontsize=9, color="#445366", va="bottom")
    fig.tight_layout(rect=(0, .13, 1, .95))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    out_of_bounds = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig_box.x1 + 1 or box.y1 > fig_box.y1 + 1:
            out_of_bounds.append(artist.get_text()[:40])
    foot_box = text.get_window_extent(renderer)
    footnote_overlaps = []
    for label in ax.get_xticklabels() + [ax.xaxis.label]:
        if label.get_text().strip() and label.get_window_extent(renderer).overlaps(foot_box):
            footnote_overlaps.append(label.get_text()[:40])
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"quantification.{ext}", dpi=200)
    plt.close(fig)
    pixels = imread(out / "quantification.png").shape
    return {"text_objects_checked_inside_figure": len(out_of_bounds) == 0,
            "out_of_bounds": out_of_bounds, "footnote_overlaps_axis_text": footnote_overlaps,
            "png_pixels_hw": [int(pixels[0]), int(pixels[1])], "figsize_in": [10, 4.2], "dpi": 200}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=REPEATS)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap resamples")
    loaded = load_run(args.run)
    analysis = analyse(loaded, args.bootstrap)
    check = check_against_metrics(args.run, analysis["models"])
    dev_n = len(loaded["views"]["tfidf"]["dev"]["y"])
    test_n = len(loaded["views"]["tfidf"]["test"]["y"])
    result = {"status": "complete", "labels": LABELS, "label_names": NAMES, "dev_n": dev_n, "test_n": test_n,
              "estimation_data": "development labels and development probabilities only; test labels score but never fit a correction",
              "method_definitions": DEFINITIONS, **analysis,
              "dev_test_diagnostics": diagnostics(loaded["views"]),
              "metrics_json_check": check, "jev_meta": loaded["jev_meta"],
              "evidence_sha256": loaded["evidence"],
              "analysis_source_sha256": {"quantify.py": digest(Path(__file__))}}
    if not args.no_figures:
        result["figure_geometry_check"] = draw(args.run, result)
    (args.run / "quantification.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    print(f"dev n={dev_n} test n={test_n} valid bootstrap replicates={analysis['bootstrap']['valid_replicates']}")
    for m in MODELS:
        for method, row in analysis["models"][m]["methods"].items():
            lo, hi = row["max_abs_share_error_pp_ci95"]
            print(f"{m:8s} {method:13s} max_abs={row['max_abs_share_error_pp']:6.2f} CI=[{lo:6.2f},{hi:6.2f}] TV={row['total_variation']:.4f}")
    for method, row in analysis["paired_jev_minus_tfidf"].items():
        lo, hi = row["ci95_jev_minus_tfidf_pp"]
        print(f"paired jev-tfidf {method:13s} {row['point_jev_minus_tfidf_pp']:+6.2f} CI=[{lo:+6.2f},{hi:+6.2f}]")


if __name__ == "__main__":
    main()
