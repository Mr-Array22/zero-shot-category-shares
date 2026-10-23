import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import io
from unittest.mock import patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from jevstudy.study import LABELS,PINNED_MODEL,digest,object_hash,request_for,validate_answer,write_jsonl
from jevstudy.metrics import bootstrap,choose_threshold,review_point,scalar_metrics
from jevstudy.prepare import phase
from jevstudy import jev_runner
def response(label=LABELS[0],tokens=100):
    return {"model":PINNED_MODEL,"answers":{"product":{"type":"choice","choice":label,"confidence":1.,
            "probabilities":{k:float(k==label) for k in LABELS}}},"usage":{"input_tokens":tokens,"output_tokens":20}}


class MetricTests(unittest.TestCase):
    def test_perfect_and_wrong_brier_scaling(self):
        y=np.array([0,1]);p=np.array([0,1]);ps=np.eye(5)[y]
        self.assertEqual(scalar_metrics(y,p,ps)["brier"],0.)
        wrong=np.eye(5)[[1,0]]
        self.assertEqual(scalar_metrics(y,np.array([1,0]),wrong)["brier"],2.)
        uniform=np.full((2,5),.2)
        self.assertAlmostEqual(scalar_metrics(y,p,uniform)["brier"],.8)

    def test_balanced_errors_can_preserve_aggregate_counts(self):
        y=np.array([0]*50+[1]*50)
        p=y.copy();p[:10]=1;p[50:60]=0
        m=scalar_metrics(y,p)
        self.assertEqual(m["accuracy"],.8)
        self.assertEqual(m["max_abs_share_error_pp"],0.)
        q=y.copy();q[:5]=1
        self.assertEqual(scalar_metrics(y,q)["accuracy"],.95)
        self.assertAlmostEqual(scalar_metrics(y,q)["max_abs_share_error_pp"],5.)

    def test_counts_reconcile_and_errors_sum_to_zero(self):
        y=np.array([0,1,2,3,4]);p=np.array([1,1,3,3,3])
        m=scalar_metrics(y,p)
        self.assertEqual(sum(m["true_counts"]),5)
        self.assertEqual(sum(m["predicted_counts"]),5)
        self.assertAlmostEqual(sum(m["share_error_pp"]),0.)

    def test_empty_accepted_set_does_not_claim_zero_error(self):
        y=np.array([0,1]);p=np.array([0,0]);ps=np.full((2,5),.2)
        point=review_point(y,p,ps,.9)
        self.assertEqual(point["coverage"],0.)
        self.assertIsNone(point["accepted_error"])
        self.assertEqual(sum(point["review_counts_by_true_class"]),2)

    def test_no_development_policy_means_review_all(self):
        y=np.array([0,1]);p=np.array([1,0]);ps=np.eye(5)[p]
        gate=choose_threshold(y,p,ps,minimum=1)
        self.assertGreater(gate["threshold"],1)

    def test_bootstrap_is_reproducible(self):
        y=np.tile(np.arange(5),10);p=y.copy();p[:5]=0
        self.assertEqual(bootstrap(y,p,repeats=100),bootstrap(y,p,repeats=100))

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            scalar_metrics(np.array([0]),np.array([0]),np.array([[.4]*5]))


class ProtocolTests(unittest.TestCase):
    def test_chronological_boundaries(self):
        self.assertEqual(phase("2016-06-30"),"train")
        self.assertEqual(phase("2016-07-01"),"dev")
        self.assertEqual(phase("2016-10-01"),"test")
        self.assertIsNone(phase("2017-01-01"))

    def test_request_excludes_reference_metadata(self):
        row={"text":"A sample narrative","id":"123","label":"mortgage","date":"2016-10-01","company":"ABC"}
        request=request_for(row)
        self.assertEqual(request["state"],{"feedback":"A sample narrative"})
        self.assertNotIn("label",request["state"])
        self.assertNotIn("company",request["state"])

    def test_heldout_words_do_not_enter_fitted_vocabulary(self):
        vectorizer=TfidfVectorizer().fit(["training mortgage payment","training credit report"])
        vectorizer.transform(["holdoutonlyword mortgage"])
        self.assertNotIn("holdoutonlyword",vectorizer.vocabulary_)

    def test_model_and_schema_validation(self):
        validate_answer(response())
        invalid=response();invalid["model"]="jev-latest"
        with self.assertRaises(ValueError):validate_answer(invalid)
        invalid=response();invalid["answers"]["product"]["probabilities"][LABELS[0]]=float("nan")
        with self.assertRaises(ValueError):validate_answer(invalid)
        invalid=response();invalid["answers"]["product"]["confidence"]=True
        with self.assertRaises(ValueError):validate_answer(invalid)

    def test_choice_near_tie_within_rounding_is_accepted(self):
        ok=response(); ps=ok["answers"]["product"]["probabilities"]
        for k in ps: ps[k]=0.0
        ps["credit_reporting"]=0.43; ps["mortgage"]=0.44; ps["debt_collection"]=0.13
        ok["answers"]["product"]["choice"]="credit_reporting"; ok["answers"]["product"]["confidence"]=0.3
        self.assertEqual(validate_answer(ok)["label"],"credit_reporting")
        bad=json.loads(json.dumps(ok)); bp=bad["answers"]["product"]["probabilities"]
        bp["credit_reporting"]=0.41; bp["mortgage"]=0.46
        with self.assertRaises(ValueError): validate_answer(bad)

    def test_probability_rounding_tolerance(self):
        def with_probabilities(values):
            value = response()
            answer = value["answers"]["product"]
            answer["probabilities"] = dict(zip(LABELS, values))
            answer["choice"] = LABELS[0]
            answer["confidence"] = (max(values) - 1 / len(LABELS)) / (1 - 1 / len(LABELS))
            return value
        for values in ([.5, .19, .1, .1, .1], [.5, .2, .1, .1, .11]):
            result = validate_answer(with_probabilities(values))
            self.assertAlmostEqual(sum(result["probabilities"]), 1, delta=1e-9)
        with self.assertRaises(ValueError):
            validate_answer(with_probabilities([.56, .2, .1, .1, .08]))



class RunnerTests(unittest.TestCase):
    def setup_run(self,folder,n=2):
        records=[{"id":str(i),"split":"dev" if i==0 else "test","request":request_for({"text":f"Synthetic schema-test message {i}"})} for i in range(n)]
        write_jsonl(folder/"requests.jsonl",records)
        (folder/"data-manifest.json").write_text(json.dumps({"requests_sha256":digest(folder/"requests.jsonl")}))

    def test_dry_run_never_reads_key_or_calls_model(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.setup_run(folder)
            with patch("jevstudy.jev_runner.getpass.getpass",side_effect=AssertionError("Key requested")):
                result=jev_runner.run(folder,execute=False,fetcher=lambda *a: self.fail("Network called"))
            self.assertEqual(result["status"],"dry_run")
            self.assertFalse((folder/"jev-responses.jsonl").exists())

    def test_budget_gate_stops_before_key_and_request(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.setup_run(folder)
            with patch("jevstudy.jev_runner.getpass.getpass",side_effect=AssertionError("Key requested")):
                result=jev_runner.run(folder,budget=.00001,execute=True,pause=0,fetcher=lambda *a:self.fail("Call made"))
            self.assertEqual(result["calls_this_invocation"],0)
            self.assertEqual(result["status"],"budget_limit")

    def test_successful_cache_is_reused_without_key(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.setup_run(folder)
            calls=[]
            def fake(body,key,timeout):
                calls.append(body);return response(),1.5
            with patch.dict(os.environ,{"TYPESAFE_API_KEY":"unit-test-value"}):
                result=jev_runner.run(folder,execute=True,pause=0,fetcher=fake)
            self.assertEqual(result["successes"],2)
            with patch("jevstudy.jev_runner.getpass.getpass",side_effect=AssertionError("Key requested")):
                again=jev_runner.run(folder,execute=True,pause=0,fetcher=lambda *a:self.fail("Cached call repeated"))
            self.assertEqual(again["calls_this_invocation"],0)
            self.assertEqual(len(calls),2)
            self.assertNotIn("unit-test-value",(folder/"jev-responses.jsonl").read_text())

    def test_failure_is_not_retried_and_keeps_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.setup_run(folder)
            def timeout(*args):raise TimeoutError("synthetic timeout")
            with patch.dict(os.environ,{"TYPESAFE_API_KEY":"unit-test-value"}):
                result=jev_runner.run(folder,execute=True,pause=0,fetcher=timeout)
            self.assertEqual(result["calls_this_invocation"],1)
            self.assertEqual(result["status"],"failed")
            self.assertGreater(result["uncertain_reserved_usd"],0)
            with self.assertRaises(ValueError):
                jev_runner.run(folder,execute=True,pause=0,fetcher=timeout)

    def test_unfinished_journal_cost_counts_against_budget(self):
        result=jev_runner.budget_used([{"event":"started","attempt_id":"x","reserved_usd":.01}])
        self.assertAlmostEqual(result["accounted_usd"],.01)
        self.assertAlmostEqual(result["known_list_price_usd"],0)

    def test_http_payload_and_response_decoding(self):
        expected=response()
        def fake_open(request,timeout):
            self.assertEqual(request.full_url,jev_runner.API_URL)
            self.assertEqual(request.method,"POST")
            self.assertEqual(json.loads(request.data),request_for({"text":"schema check"}))
            self.assertEqual(request.get_header("Authorization"),"Bearer unit-test-value")
            return io.BytesIO(json.dumps(expected).encode())
        with patch("jevstudy.jev_runner.urllib.request.urlopen",side_effect=fake_open):
            actual,latency=jev_runner.fetch(request_for({"text":"schema check"}),"unit-test-value",30)
        self.assertEqual(actual,expected)
        self.assertGreaterEqual(latency,0)


if __name__=="__main__":unittest.main()
