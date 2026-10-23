import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import zlib
from unittest.mock import patch

import numpy as np

from jevstudy import embed_baseline
from jevstudy.study import LABELS, digest, object_hash, read_jsonl, write_jsonl

DIM = 256
KEYWORDS = {
    "bank_account": "checking savings deposit",
    "credit_card": "card purchase balance",
    "credit_reporting": "report bureau inquiry",
    "debt_collection": "collector collect debt",
    "mortgage": "mortgage escrow foreclosure",
}


def stub_encoder(texts):
    """Deterministic hashed bag-of-words encoder. Needs no model download."""
    vectors = np.zeros((len(texts), DIM))
    for i, text in enumerate(texts):
        for token in re.findall(r"[a-z]+", text.lower()):
            vectors[i, zlib.crc32(token.encode("utf-8")) % DIM] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def synthetic_records():
    counts = {"train": 10, "dev": 3, "test": 4}
    records = []
    n = 0
    for split, per_label in counts.items():
        for label in LABELS:
            for k in range(per_label):
                n += 1
                text = f"complaint {k} about {KEYWORDS[label]} account note {n}"
                records.append({"id": f"{split}-{n}", "date": "2016-01-01", "split": split,
                                "label": label, "text": text})
    return records


class EmbedBaselineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.records = synthetic_records()
        self._write_study(self.records)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_study(self, records, manifest_hash=None):
        write_jsonl(self.folder / "study.jsonl", records)
        digest_value = manifest_hash or digest(self.folder / "study.jsonl")
        (self.folder / "data-manifest.json").write_text(json.dumps({"study_sha256": digest_value}))

    def test_predictions_match_baseline_schema(self):
        embed_baseline.run(self.folder, encoder=stub_encoder, repeats=20)
        rows = read_jsonl(self.folder / "embed-predictions.jsonl")
        texts = {r["id"]: r["text"] for r in self.records}
        evaluation = [r for r in self.records if r["split"] != "train"]
        self.assertEqual(len(rows), len(evaluation))
        self.assertEqual({r["id"] for r in rows}, {r["id"] for r in evaluation})
        for row in rows:
            self.assertEqual(set(row), {"id", "split", "method", "label", "probabilities", "latency_ms", "input_hash"})
            self.assertEqual(row["method"], "embed_minilm")
            self.assertEqual(len(row["probabilities"]), len(LABELS))
            self.assertAlmostEqual(sum(row["probabilities"]), 1.0, places=6)
            self.assertEqual(row["label"], LABELS[int(np.argmax(row["probabilities"]))])
            self.assertEqual(row["input_hash"], object_hash(texts[row["id"]]))
            self.assertGreaterEqual(row["latency_ms"], 0)

    def test_metrics_file_has_required_fields(self):
        embed_baseline.run(self.folder, encoder=stub_encoder, repeats=20)
        metrics = json.loads((self.folder / "embed-metrics.json").read_text())
        self.assertGreaterEqual(metrics["test"]["accuracy"], 0.9)
        for key in ("accuracy", "macro_f1", "max_abs_share_error_pp", "brier", "share_error_pp"):
            self.assertIn(key, metrics["test"])
        self.assertEqual(set(metrics["test"]["ci95_percentile_bootstrap"]),
                         {"accuracy", "macro_f1", "max_abs_share_error_pp", "total_variation", "brier"})
        self.assertEqual(metrics["bootstrap_seed"], 20261022)
        self.assertEqual(metrics["bootstrap_repeats"], 20)
        self.assertIsNone(metrics["paired_vs_tfidf_test"])

    def test_injected_encoder_skips_model_loading(self):
        with patch.object(embed_baseline, "load_encoder", side_effect=AssertionError("must not load")):
            embed_baseline.run(self.folder, encoder=stub_encoder, repeats=5)

    def test_library_is_only_needed_when_loading_a_real_encoder(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(ImportError):
                embed_baseline.load_encoder()

    def test_rejects_changed_study_hash(self):
        self._write_study(self.records, manifest_hash="0" * 64)
        with self.assertRaises(ValueError):
            embed_baseline.run(self.folder, encoder=stub_encoder, repeats=5)

    def test_rejects_training_set_missing_a_category(self):
        records = [r for r in self.records if not (r["split"] == "train" and r["label"] == "mortgage")]
        self._write_study(records)
        with self.assertRaises(ValueError):
            embed_baseline.run(self.folder, encoder=stub_encoder, repeats=5)

    def test_paired_comparison_when_tfidf_predictions_exist(self):
        tfidf = []
        for r in self.records:
            if r["split"] == "train":
                continue
            probs = [0.0] * len(LABELS)
            probs[LABELS.index(r["label"])] = 1.0
            tfidf.append({"id": r["id"], "split": r["split"], "method": "tfidf", "label": r["label"],
                          "probabilities": probs, "latency_ms": 0.1, "input_hash": object_hash(r["text"])})
        write_jsonl(self.folder / "local-predictions.jsonl", tfidf)
        embed_baseline.run(self.folder, encoder=stub_encoder, repeats=20)
        metrics = json.loads((self.folder / "embed-metrics.json").read_text())
        paired = metrics["paired_vs_tfidf_test"]
        self.assertIsNotNone(paired)
        for name in ("macro_f1", "accuracy", "max_abs_share_error_pp"):
            self.assertEqual(len(paired["embed_minus_tfidf"][name]["ci95"]), 2)
        self.assertLessEqual(paired["embed_minus_tfidf"]["accuracy"]["point"], 0.0)

    def test_stub_encoder_is_deterministic(self):
        first = stub_encoder(["a bank account", "mortgage escrow"])
        second = stub_encoder(["a bank account", "mortgage escrow"])
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
