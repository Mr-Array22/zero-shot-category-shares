#!/usr/bin/env python3
"""Sentence-embedding baseline: frozen encoder features plus logistic regression.

Uses the same frozen study text, split rules, label order, classifier settings,
input hashes and metrics as baselines.py. The encoder can be injected; the
sentence-transformers package is imported only when no encoder is passed, so this
module and its tests run without it. For a real run, install requirements-embed.txt.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .metrics import bootstrap, scalar_metrics
from .study import LABELS, digest, object_hash, read_jsonl, write_jsonl

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METHOD = "embed_minilm"
SEED = 20261022
BOOTSTRAP_REPEATS = 500


def load_encoder(model_name=DEFAULT_MODEL):
    """Return encode(texts) -> float array of shape (n, d). Imports the library lazily."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def encode(texts):
        vectors = model.encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=float)

    encode.info = {
        "library": "sentence-transformers",
        "library_version": importlib.metadata.version("sentence-transformers"),
        "model": model_name,
        "max_seq_length": int(model.max_seq_length),
        "embedding_dimension": int(model.get_sentence_embedding_dimension()),
        "normalize_embeddings": True,
        "truncation_note": "The encoder truncates each input to max_seq_length tokens; the classifiers for other methods see the full text.",
    }
    return encode


def _arrays(rows, predictions):
    truth = np.array([LABELS.index(r["label"]) for r in rows])
    pred = np.array([LABELS.index(predictions[r["id"]]["label"]) for r in rows])
    probabilities = np.array([predictions[r["id"]]["probabilities"] for r in rows], dtype=float)
    return truth, pred, probabilities


def _summary(rows, predictions, repeats):
    truth, pred, probabilities = _arrays(rows, predictions)
    result = scalar_metrics(truth, pred, probabilities)
    result["ci95_percentile_bootstrap"] = bootstrap(truth, pred, probabilities, repeats=repeats, seed=SEED)
    timings = np.array([predictions[r["id"]]["latency_ms"] for r in rows])
    result["latency_ms"] = {"mean": float(timings.mean()), "p50": float(np.quantile(timings, .5)),
                            "p95": float(np.quantile(timings, .95)), "n": len(rows)}
    return result


def _paired_vs_tfidf(rows, predictions, folder, repeats):
    path = folder / "local-predictions.jsonl"
    if not path.exists():
        return None
    tfidf = {r["id"]: r for r in read_jsonl(path) if r["method"] == "tfidf"}
    missing = {r["id"] for r in rows} - set(tfidf)
    if missing:
        raise ValueError(f"TF-IDF predictions are missing {len(missing)} test records")
    truth, pred, _ = _arrays(rows, predictions)
    _, other, _ = _arrays(rows, tfidf)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(repeats):
        ix = rng.integers(0, len(rows), len(rows))
        a = scalar_metrics(truth[ix], pred[ix])
        b = scalar_metrics(truth[ix], other[ix])
        differences.append([a["macro_f1"] - b["macro_f1"], a["accuracy"] - b["accuracy"],
                            a["max_abs_share_error_pp"] - b["max_abs_share_error_pp"]])
    diffs = np.array(differences)
    point_a = scalar_metrics(truth, pred)
    point_b = scalar_metrics(truth, other)
    names = ["macro_f1", "accuracy", "max_abs_share_error_pp"]
    return {
        "unit": "paired test complaints; the same resample indices are used for both methods",
        "bootstrap_repeats": repeats,
        "bootstrap_seed": SEED,
        "embed_minus_tfidf": {
            name: {"point": point_a[name] - point_b[name],
                   "ci95": np.quantile(diffs[:, i], [.025, .975]).tolist()}
            for i, name in enumerate(names)
        },
    }


def run(folder, model_name=DEFAULT_MODEL, encoder=None, repeats=BOOTSTRAP_REPEATS):
    folder = Path(folder)
    manifest = json.loads((folder / "data-manifest.json").read_text())
    study_path = folder / "study.jsonl"
    if digest(study_path) != manifest["study_sha256"]:
        raise ValueError("Study data hash changed")
    records = read_jsonl(study_path)
    train = [r for r in records if r["split"] == "train"]
    evaluation = [r for r in records if r["split"] != "train"]
    if set(r["label"] for r in train) != set(LABELS):
        raise ValueError("Training set must contain every category")
    if encoder is None:
        encoder = load_encoder(model_name)

    start = time.perf_counter()
    train_vectors = encoder([r["text"] for r in train])
    embed_train_seconds = time.perf_counter() - start

    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1500, random_state=SEED)
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train_vectors, [r["label"] for r in train])
    fit_seconds = time.perf_counter() - start

    indices = [list(classifier.classes_).index(label) for label in LABELS]
    outputs = []
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for row in evaluation:
            tick = time.perf_counter()
            vector = encoder([row["text"]])
            probabilities = classifier.predict_proba(vector)[0][indices].tolist()
            label = LABELS[int(np.argmax(probabilities))]
            latency = (time.perf_counter() - tick) * 1000
            outputs.append({"id": row["id"], "split": row["split"], "method": METHOD, "label": label,
                            "probabilities": probabilities, "latency_ms": latency,
                            "input_hash": object_hash(row["text"])})
    evaluation_seconds = time.perf_counter() - start

    predictions_path = folder / "embed-predictions.jsonl"
    write_jsonl(predictions_path, outputs)
    predictions = {r["id"]: r for r in outputs}
    test = [r for r in evaluation if r["split"] == "test"]
    dev = [r for r in evaluation if r["split"] == "dev"]
    result = {
        "method": METHOD,
        "model": model_name,
        "encoder": getattr(encoder, "info", {"library": "injected_callable", "model": model_name}),
        "study_sha256": manifest["study_sha256"],
        "predictions_sha256": digest(predictions_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "threadpool_limit": 1,
        "embed_train_seconds": embed_train_seconds,
        "fit_seconds": fit_seconds,
        "evaluation_wall_seconds": evaluation_seconds,
        "classifier": {"C": 1.0, "max_iter": 1500, "solver": "lbfgs", "random_state": SEED,
                       "iterations": classifier.n_iter_.tolist(), "features": int(train_vectors.shape[1])},
        "timing_design": "Single records; encoder and classifier stay in memory; no batching; training-set embedding and fit are excluded. Local CPU timing is not hosted-model compute latency.",
        "bootstrap_repeats": repeats,
        "bootstrap_seed": SEED,
        "test": _summary(test, predictions, repeats),
        "dev": _summary(dev, predictions, repeats),
        "paired_vs_tfidf_test": _paired_vs_tfidf(test, predictions, folder, repeats),
        "source_sha256": {p.name: digest(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
    }
    (folder / "embed-metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = run(args.run, model_name=args.model)
    print(json.dumps({
        "test_accuracy": result["test"]["accuracy"],
        "test_macro_f1": result["test"]["macro_f1"],
        "test_max_abs_share_error_pp": result["test"]["max_abs_share_error_pp"],
        "test_brier": result["test"]["brier"],
        "fit_seconds": round(result["fit_seconds"], 3),
        "saved": [str(args.run / "embed-predictions.jsonl"), str(args.run / "embed-metrics.json")],
    }, indent=2))


if __name__ == "__main__":
    main()
