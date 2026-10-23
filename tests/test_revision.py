import csv
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from jevstudy import prepare
from jevstudy import revision
from jevstudy.metrics import scalar_metrics
from jevstudy.study import CRITERIA, INSTRUCTIONS, LABELS, PINNED_MODEL, digest, object_hash, read_jsonl, request_for, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "main"


def response(label=LABELS[0]):
    return {"model": PINNED_MODEL, "answers": {"product": {"type": "choice", "choice": label, "confidence": 1.,
            "probabilities": {k: float(k == label) for k in LABELS}}},
            "usage": {"input_tokens": 100, "output_tokens": 20}}


def write_definitions(path, criteria=None):
    data = {"instructions": INSTRUCTIONS, "criteria": dict(criteria or CRITERIA)}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def make_zip(path, rows):
    header = ["Complaint ID", "Date received", "Product", "Consumer complaint narrative", "Company"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rows.csv", buffer.getvalue())
    return path


OLD = ["Bank account or service", "Credit card", "Credit reporting", "Debt collection", "Mortgage"]


def fixture_rows():
    rows = []
    dates = ["01/10/2017", "02/10/2017", "03/10/2017", "04/10/2017", "04/23/2017"]
    for i in range(40):
        rows.append([str(1000 + i), dates[i % 5], OLD[i % 5],
                     f"Synthetic complaint number {i} describes a problem with the account and the response.", "ACME"])
    rows.append(["2000", "04/24/2017", OLD[0], "Post-change record that must be excluded by date window.", "ACME"])
    rows.append(["2001", "12/20/2016", OLD[1], "A late 2016 record that belongs only to the unused-2016q4 window.", "ACME"])
    rows.append(["2002", "12/21/2016", OLD[2], "Another late 2016 record used for the unused-2016q4 window.", "ACME"])
    rows.append(["2003", "12/22/2016", OLD[3], "Third late 2016 record; its id is in the study exclusion list.", "ACME"])
    rows.append(["2004", "02/01/2017", "Checking or savings account", "Renamed product after the taxonomy change.", "ACME"])
    rows.append(["2005", "02/02/2017", OLD[4], "too short", "ACME"])
    rows.append(["2006", "02/03/2017", OLD[0], "Shared narrative text that appears twice with conflicting labels here.", "ACME"])
    rows.append(["2007", "02/04/2017", OLD[3], "Shared narrative text that appears twice with conflicting labels here.", "ACME"])
    return rows


class V1ReproductionTests(unittest.TestCase):
    @unittest.skipUnless((RUN / "requests.jsonl").exists(), "completed run folder not present")
    def test_v1_requests_rebuild_byte_for_byte(self):
        manifest = json.loads((RUN / "data-manifest.json").read_text())
        records = [r for r in read_jsonl(RUN / "study.jsonl") if r["split"] != "train"]
        rebuilt = [{"id": r["id"], "split": r["split"], "request": request_for(r)} for r in records]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "requests.jsonl"
            write_jsonl(path, rebuilt)
            self.assertEqual(digest(path), manifest["requests_sha256"])

    def test_v1_definitions_file_matches_current_criteria(self):
        v1 = json.loads((ROOT / "definitions" / "v1.json").read_text())
        self.assertEqual(v1["criteria"], CRITERIA)
        self.assertEqual(v1["instructions"], INSTRUCTIONS)

    def test_v2_changes_only_the_three_dev_driven_categories(self):
        v2 = revision.load_definitions(ROOT / "definitions" / "v2.json")
        changed = sorted(k for k in LABELS if v2["criteria"][k] != CRITERIA[k])
        self.assertEqual(changed, ["credit_card", "credit_reporting", "debt_collection"])
        self.assertEqual(v2["instructions"], INSTRUCTIONS)


class FreshPrepareTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.zip = make_zip(self.dir / "archive.zip", fixture_rows())
        text = "Third late 2016 record; its id is in the study exclusion list."
        self.study = self.dir / "study.jsonl"
        write_jsonl(self.study, [
            {"id": "1005", "split": "test", "group_hash": "x", "label": "mortgage", "text": "unrelated"},
            {"id": "9999", "split": "train", "group_hash": object_hash(text.lower()), "label": "mortgage", "text": text},
        ])

    def test_2017q1_window_excludes_post_change_and_seen_rows(self):
        out = self.dir / "fresh"
        prepare.prepare_fresh(self.zip, out, "2017q1", self.study, n=10)
        rows = read_jsonl(out / "study.jsonl")
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r["split"] == "fresh" for r in rows))
        self.assertTrue(all("2017-01-01" <= r["date"] <= "2017-04-23" for r in rows))
        manifest = json.loads((out / "data-manifest.json").read_text())
        self.assertEqual(manifest["window_end_exclusive"], "2017-04-24")
        self.assertEqual(manifest["filter_counts"]["excluded_date"], 4)
        self.assertEqual(manifest["filter_counts"]["excluded_product"], 1)
        self.assertEqual(manifest["filter_counts"]["conflicting_input_groups_removed"], 1)
        self.assertEqual(manifest["requests_sha256"], digest(out / "requests.jsonl"))
        self.assertEqual(manifest["study_sha256"], digest(out / "study.jsonl"))
        self.assertTrue(manifest["fresh"])

    def test_sampling_is_deterministic_and_label_blind(self):
        a, b = self.dir / "a", self.dir / "b"
        prepare.prepare_fresh(self.zip, a, "2017q1", self.study, n=10)
        prepare.prepare_fresh(self.zip, b, "2017q1", self.study, n=10)
        self.assertEqual(digest(a / "study.jsonl"), digest(b / "study.jsonl"))

    def test_unused_2016q4_window_excludes_study_ids_and_texts(self):
        out = self.dir / "fresh2"
        prepare.prepare_fresh(self.zip, out, "unused-2016q4", self.study, n=2)
        rows = read_jsonl(out / "study.jsonl")
        self.assertTrue(all("2016-10-01" <= r["date"] < "2017-01-01" for r in rows))
        self.assertNotIn("2003", {r["id"] for r in rows})

    def test_requests_are_v1_and_pinned(self):
        out = self.dir / "fresh3"
        prepare.prepare_fresh(self.zip, out, "2017q1", self.study, n=5)
        requests = read_jsonl(out / "requests.jsonl")
        self.assertEqual(len(requests), 5)
        self.assertTrue(all(r["request"]["model"] == PINNED_MODEL for r in requests))
        self.assertTrue(all(r["request"]["questions"]["product"]["criteria"] == CRITERIA for r in requests))

    def test_refuses_too_large_and_non_empty(self):
        with self.assertRaises(ValueError):
            prepare.prepare_fresh(self.zip, self.dir / "big", "2017q1", self.study, n=10000)
        out = self.dir / "used"
        out.mkdir()
        (out / "x.txt").write_text("x")
        with self.assertRaises(ValueError):
            prepare.prepare_fresh(self.zip, out, "2017q1", self.study, n=2)


class DevAndRunnerTests(unittest.TestCase):
    def make_source(self, folder):
        rows = []
        for split, count in (("train", 10), ("dev", 6), ("test", 6)):
            for i in range(count):
                label = LABELS[i % 5]
                rows.append({"id": f"{split}{i}", "split": split, "date": "2016-08-01", "label": label,
                             "text": f"Synthetic {split} narrative {i} about {label} and a credit report.",
                             "group_hash": f"{split}-{i}"})
        write_jsonl(folder / "study.jsonl", rows)
        requests = [{"id": r["id"], "split": r["split"], "request": request_for(r)} for r in rows if r["split"] != "train"]
        write_jsonl(folder / "requests.jsonl", requests)
        (folder / "data-manifest.json").write_text(json.dumps({"study_sha256": digest(folder / "study.jsonl"),
                                                              "requests_sha256": digest(folder / "requests.jsonl")}))
        return rows

    def test_build_dev_v2_dry_run_writes_nothing_and_counts_dev(self):
        src = Path(tempfile.mkdtemp())
        self.make_source(src)
        out = src / "dev-v2"
        summary = revision.build_dev_v2(src, out, dry_run=True)
        self.assertEqual(summary["requests"], 6)
        self.assertEqual(summary["model_calls"], 0)
        self.assertFalse(out.exists())

    def test_build_dev_v2_uses_v2_criteria_and_pinned_model(self):
        src = Path(tempfile.mkdtemp())
        self.make_source(src)
        out = src / "dev-v2"
        revision.build_dev_v2(src, out)
        requests = read_jsonl(out / "requests.jsonl")
        v2 = revision.load_definitions(ROOT / "definitions" / "v2.json")
        self.assertEqual(len(requests), 6)
        for r in requests:
            self.assertEqual(r["request"]["model"], PINNED_MODEL)
            self.assertEqual(r["request"]["questions"]["product"]["criteria"], v2["criteria"])
            self.assertEqual(r["request"]["state"], {"feedback": r["request"]["state"]["feedback"]})

    def test_run_is_zero_calls_without_execute(self):
        src = Path(tempfile.mkdtemp())
        self.make_source(src)
        out = src / "dev-v2"
        revision.build_dev_v2(src, out)
        result = revision.run_arm(out, execute=False, fetcher=lambda *a: self.fail("network called"))
        self.assertEqual(result["status"], "dry_run")

    def test_run_with_fake_fetcher_saves_responses_without_key_in_files(self):
        src = Path(tempfile.mkdtemp())
        self.make_source(src)
        out = src / "dev-v2"
        revision.build_dev_v2(src, out)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "unit-test-value"}):
            result = revision.run_arm(out, execute=True, pause=0, fetcher=lambda body, key, timeout: (response(), 1.0))
        self.assertEqual(result["successes"], 6)
        self.assertNotIn("unit-test-value", (out / "jev-responses.jsonl").read_text())

    def test_eval_dev_counts_debt_to_credit_reporting(self):
        src = Path(tempfile.mkdtemp())
        rows = self.make_source(src)
        dev = [r for r in rows if r["split"] == "dev"]
        v1 = src / "v1"
        v1.mkdir()
        write_jsonl(v1 / "study.jsonl", rows)
        (v1 / "data-manifest.json").write_text(json.dumps({"study_sha256": digest(v1 / "study.jsonl")}))
        write_jsonl(v1 / "jev-responses.jsonl", [{"id": r["id"], "split": "dev", "status": "ok",
                    "response": response(LABELS[2] if r["label"] == "debt_collection" else r["label"])} for r in dev])
        out = src / "dev-v2"
        revision.build_dev_v2(src, out)
        write_jsonl(out / "jev-responses.jsonl", [{"id": r["id"], "split": "dev", "status": "ok",
                    "response": response(r["label"])} for r in dev])
        result = revision.eval_dev(v1, out)
        self.assertEqual(result["v1"]["debt_collection_to_credit_reporting"], sum(r["label"] == "debt_collection" for r in dev))
        self.assertEqual(result["v2"]["accuracy"], 1.0)
        self.assertEqual(result["v2_minus_v1"]["accuracy"], 1.0 - result["v1"]["accuracy"])
        self.assertTrue((out / "dev-metrics.json").exists())


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.defs = write_definitions(self.dir / "v2.json")
        self.prereg = self.dir / "PREREGISTRATION.md"
        self.prereg.write_text("# Prereg\n\n<!-- FROZEN:BEGIN -->\n(empty until freeze)\n<!-- FROZEN:END -->\n", encoding="utf-8")

    def test_unfrozen_refuses_and_freeze_then_verify(self):
        with self.assertRaises(ValueError):
            revision.verify_frozen(self.defs, self.prereg)
        state = revision.freeze(self.defs, self.prereg)
        self.assertEqual(state["definitions_sha256"], digest(self.defs))
        self.assertEqual(revision.verify_frozen(self.defs, self.prereg)["definitions_sha256"], digest(self.defs))
        self.assertEqual(revision.freeze(self.defs, self.prereg)["definitions_sha256"], digest(self.defs))

    def test_changed_definitions_are_refused_after_freeze(self):
        revision.freeze(self.defs, self.prereg)
        write_definitions(self.defs, {**CRITERIA, "mortgage": "Changed after freezing."})
        with self.assertRaises(ValueError):
            revision.verify_frozen(self.defs, self.prereg)
        with self.assertRaises(ValueError):
            revision.freeze(self.defs, self.prereg)

    def test_build_fresh_requires_freeze_and_builds_both_arms(self):
        dir = self.dir
        zipped = make_zip(dir / "archive.zip", fixture_rows())
        study = dir / "orig.jsonl"
        write_jsonl(study, [{"id": "0", "group_hash": "none", "split": "train"}])
        fresh = dir / "fresh"
        prepare.prepare_fresh(zipped, fresh, "2017q1", study, n=6)
        with self.assertRaises(ValueError):
            revision.build_fresh(fresh, self.defs, self.prereg)
        revision.freeze(self.defs, self.prereg)
        summaries = revision.build_fresh(fresh, self.defs, self.prereg)
        self.assertEqual(summaries["v1"]["requests"], 6)
        self.assertEqual(summaries["v2"]["requests"], 6)
        v2_manifest = json.loads((fresh / "jev-v2" / "data-manifest.json").read_text())
        self.assertTrue(v2_manifest["preregistered"])
        self.assertEqual(v2_manifest["definitions_sha256"], digest(self.defs))
        # A frozen-arm run refuses if the definitions file has changed.
        write_definitions(self.defs, {**CRITERIA, "mortgage": "Changed after freezing."})
        with self.assertRaises(ValueError):
            revision.run_arm(fresh / "jev-v2", execute=False, v2_path=self.defs, prereg=self.prereg)


class RepeatAndMetricTests(unittest.TestCase):
    def test_repeat_requests_are_identical_to_originals(self):
        src = Path(tempfile.mkdtemp())
        rows = []
        for i in range(8):
            label = LABELS[i % 5]
            rows.append({"id": f"{2000+i}", "split": "test", "date": f"2016-10-{i+1:02d}", "label": label,
                         "text": f"Synthetic test narrative {i} with enough characters to be a request.", "group_hash": str(i)})
        write_jsonl(src / "study.jsonl", rows)
        write_jsonl(src / "requests.jsonl", [{"id": r["id"], "split": "test", "request": request_for(r)} for r in rows])
        (src / "data-manifest.json").write_text(json.dumps({"study_sha256": digest(src / "study.jsonl")}))
        out = src / "repeat"
        revision.build_repeat(src, out, n=3, copies=2)
        requests = read_jsonl(out / "requests.jsonl")
        self.assertEqual(len(requests), 6)
        self.assertEqual(len({r["id"] for r in requests}), 6)
        originals = {r["id"]: r["request"] for r in read_jsonl(src / "requests.jsonl")}
        for r in requests:
            original_id = r["id"].split("#")[0]
            self.assertEqual(r["request"], originals[original_id])

    def test_eval_repeat_reports_agreement(self):
        src = Path(tempfile.mkdtemp())
        rows = [{"id": "1", "split": "test", "date": "2016-10-01", "label": LABELS[0], "text": "x" * 50, "group_hash": "a"}]
        write_jsonl(src / "study.jsonl", rows)
        write_jsonl(src / "requests.jsonl", [{"id": "1", "split": "test", "request": request_for(rows[0])}])
        (src / "data-manifest.json").write_text(json.dumps({"study_sha256": digest(src / "study.jsonl")}))
        write_jsonl(src / "jev-responses.jsonl", [{"id": "1", "split": "test", "status": "ok", "response": response(LABELS[0])}])
        out = src / "repeat"
        revision.build_repeat(src, out, n=1, copies=2)
        write_jsonl(out / "jev-responses.jsonl", [
            {"id": "1#r1", "split": "repeat", "status": "ok", "response": response(LABELS[0])},
            {"id": "1#r2", "split": "repeat", "status": "ok", "response": response(LABELS[1])}])
        result = revision.eval_repeat(out, src)
        self.assertEqual(result["label_agreement"], 0.5)
        self.assertEqual(result["label_disagreements"], 1)

    def test_paired_bootstrap_is_reproducible_and_zero_for_identical_arms(self):
        truth = np.tile(np.arange(5), 20)
        pred = truth.copy()
        pred[:10] = 0
        arms = {"a": (pred, None), "b": (pred.copy(), None)}
        first = revision.bootstrap_arms(truth, arms, repeats=50)
        second = revision.bootstrap_arms(truth, arms, repeats=50)
        self.assertEqual(first, second)
        diff = np.array(first["a"]["accuracy"]) - np.array(first["b"]["accuracy"])
        self.assertTrue(np.all(diff == 0))

    def test_metric_row_matches_scalar_metrics(self):
        truth = np.array([0, 1, 2, 3, 4])
        pred = np.array([0, 1, 2, 3, 3])
        row = revision.metric_row(scalar_metrics(truth, pred))
        self.assertAlmostEqual(row["accuracy"], 0.8)
        self.assertAlmostEqual(row["credit_reporting_share_error_pp"], 0.0)


class EvalFreshTests(unittest.TestCase):
    def test_eval_fresh_end_to_end_without_network(self):
        d = Path(tempfile.mkdtemp())
        train_src = d / "train-source"
        train_src.mkdir()
        rows = []
        for split, count in (("train", 60), ("dev", 10), ("test", 10)):
            for i in range(count):
                label = LABELS[i % 5]
                rows.append({"id": f"{split}{i}", "split": split, "date": "2016-08-01", "label": label,
                             "text": f"{label} account complaint about records {label} number {i}. The company reported the {label} issue again.",
                             "group_hash": f"{split}-{i}"})
        write_jsonl(train_src / "study.jsonl", rows)
        write_jsonl(train_src / "requests.jsonl", [{"id": r["id"], "split": r["split"], "request": request_for(r)} for r in rows if r["split"] != "train"])
        (train_src / "data-manifest.json").write_text(json.dumps({"study_sha256": digest(train_src / "study.jsonl"),
                                                                  "requests_sha256": digest(train_src / "requests.jsonl")}))
        pipeline, indices = revision.fit_tfidf([r for r in rows if r["split"] == "train"])
        evaluation = [r for r in rows if r["split"] != "train"]
        labels, _ = revision.predict_tfidf(pipeline, indices, evaluation)
        write_jsonl(train_src / "local-predictions.jsonl", [{"id": r["id"], "split": r["split"], "method": "tfidf",
                    "label": LABELS[int(l)]} for r, l in zip(evaluation, labels)])

        fresh = d / "fresh"
        fresh.mkdir()
        fresh_rows = [{**r, "id": f"f{i}", "split": "fresh", "date": "2017-02-01"} for i, r in enumerate(rows[:20])]
        write_jsonl(fresh / "study.jsonl", fresh_rows)
        write_jsonl(fresh / "requests.jsonl", [{"id": r["id"], "split": "fresh", "request": request_for(r)} for r in fresh_rows])
        (fresh / "data-manifest.json").write_text(json.dumps({"fresh": True, "fresh_window": "2017q1",
            "window_start": "2017-01-01", "window_end_exclusive": "2017-04-24",
            "study_sha256": digest(fresh / "study.jsonl"), "requests_sha256": digest(fresh / "requests.jsonl")}))
        defs = write_definitions(d / "v2.json")
        prereg = d / "PREREGISTRATION.md"
        prereg.write_text("<!-- FROZEN:BEGIN -->\n(empty)\n<!-- FROZEN:END -->\n", encoding="utf-8")
        revision.freeze(defs, prereg)
        revision.build_fresh(fresh, defs, prereg)
        write_jsonl(fresh / "jev-v1" / "jev-responses.jsonl", [{"id": r["id"], "split": "fresh", "status": "ok",
                    "response": response(r["label"])} for r in fresh_rows])
        write_jsonl(fresh / "jev-v2" / "jev-responses.jsonl", [{"id": r["id"], "split": "fresh", "status": "ok",
                    "response": response(LABELS[0])} for r in fresh_rows])
        result = revision.eval_fresh(fresh, train_src, defs, prereg)
        self.assertEqual(result["tfidf_reproduction"]["matches"], len(evaluation))
        self.assertAlmostEqual(result["arms"]["jev_v1"]["accuracy"], 1.0)
        self.assertIn("fixes_bias_claim", result["decision"])
        self.assertTrue((fresh / "fresh-metrics.json").exists())
        self.assertEqual(len(result["bootstrap_ci95"]["jev_v2"]["accuracy"]), 2)



if __name__ == "__main__":
    unittest.main()
