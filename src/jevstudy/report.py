#!/usr/bin/env python3
"""Evaluate a run folder: metrics, comparison table, review policy, error list and charts."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .metrics import bootstrap,choose_threshold,confusion,reliability,review_point,scalar_metrics
from .study import LABELS,NAMES,PINNED_MODEL,digest,object_hash,read_jsonl,validate_answer

ROOT=Path(__file__).resolve().parent  # package directory
DISPLAY={"keyword":"Keyword rules","tfidf":"TF-IDF + logistic regression","jev":"Jev 1.14"}


def arrays(rows,predictions):
    truth=np.array([LABELS.index(r["label"]) for r in rows])
    pred=np.array([LABELS.index(predictions[r["id"]]["label"]) for r in rows])
    raw=[predictions[r["id"]]["probabilities"] for r in rows]
    probs=np.array(raw) if all(p is not None for p in raw) else None
    return truth,pred,probs


def build(folder,repeats=500):
    data=json.loads((folder/"data-manifest.json").read_text())
    if digest(folder/"study.jsonl")!=data["study_sha256"] or digest(folder/"requests.jsonl")!=data["requests_sha256"]:
        raise ValueError("Frozen study/request hash mismatch")
    rows=read_jsonl(folder/"study.jsonl")
    by_id={r["id"]:r for r in rows}
    dev=[r for r in rows if r["split"]=="dev"]
    test=[r for r in rows if r["split"]=="test"]
    methods={"keyword":{},"tfidf":{}}
    local=json.loads((folder/"local-manifest.json").read_text())
    if digest(folder/"local-predictions.jsonl")!=local["predictions_sha256"] or local["study_sha256"]!=data["study_sha256"]:
        raise ValueError("Local predictions do not match frozen study")
    for r in read_jsonl(folder/"local-predictions.jsonl"):
        if r["id"] in methods[r["method"]] or r["id"] not in by_id or r["input_hash"]!=object_hash(by_id[r["id"]]["text"]):
            raise ValueError("Invalid/duplicate local prediction or input hash")
        methods[r["method"]][r["id"]]=r
    expected={r["id"] for r in dev+test}
    for name in methods:
        if set(methods[name])!=expected:
            raise ValueError(f"Incomplete local method {name}")
    successes={}
    attempted=[]
    jev_status={"status":"not_run","successful":0,"expected":len(expected),"failures":0,"inference_calls_here":0}
    path=folder/"jev-responses.jsonl"
    if path.exists():
        requests={r["id"]:r for r in read_jsonl(folder/"requests.jsonl")}
        attempted=read_jsonl(path)
        for r in attempted:
            if r["status"]!="ok":
                continue
            if r["id"] in successes or r["id"] not in requests or r["request_hash"]!=object_hash(requests[r["id"]]["request"]):
                raise ValueError("Duplicate or mismatched Jev response")
            answer=validate_answer(r["response"])
            successes[r["id"]]={**answer,"latency_ms":r["latency_ms"]}
        jev_status={"status":"complete" if set(successes)==expected else "partial",
                    "successful":len(successes),"expected":len(expected),"failures":sum(r["status"]!="ok" for r in attempted),
                    "inference_calls_here":0,"note":"Report generation is offline. Partial Jev results are not scored against complete baselines."}
        if jev_status["status"]=="complete":
            methods["jev"]=successes
    metrics={}
    for name,predictions in methods.items():
        y,p,ps=arrays(test,predictions)
        result=scalar_metrics(y,p,ps)
        result["ci95_percentile_bootstrap"]=bootstrap(y,p,ps,repeats)
        timings=np.array([predictions[r["id"]]["latency_ms"] for r in test])
        if not np.isfinite(timings).all() or (timings<0).any():
            raise ValueError("Invalid timing value")
        result["latency_ms"]={"mean":float(timings.mean()),"p50":float(np.quantile(timings,.5)),"p95":float(np.quantile(timings,.95)),"n":len(test)}
        if ps is not None:
            yd,pd,pdev=arrays(dev,predictions)
            gate=choose_threshold(yd,pd,pdev,minimum=min(100,len(dev)))
            result["review_policy"]=gate
            result["test_review_point"]=review_point(y,p,ps,gate["threshold"])
            result["review_curve"]= [review_point(y,p,ps,float(t)) for t in np.linspace(0,1,51)]
            result["reliability"]=reliability(y,p,ps)
        metrics[name]=result
    paired={}
    for challenger in ("keyword","jev"):
        if challenger not in methods:
            continue
        ya,pa,psa=arrays(test,methods[challenger]);_,pb,psb=arrays(test,methods["tfidf"])
        rng=np.random.default_rng(20261022)
        differences=[]
        for _ in range(repeats):
            ix=rng.integers(0,len(test),len(test))
            a=scalar_metrics(ya[ix],pa[ix]);b=scalar_metrics(ya[ix],pb[ix])
            differences.append([a["macro_f1"]-b["macro_f1"],a["max_abs_share_error_pp"]-b["max_abs_share_error_pp"]])
        paired[challenger+"_minus_tfidf"]={"macro_f1_ci95":np.quantile(np.array(differences)[:,0],[.025,.975]).tolist(),
                                            "max_abs_share_error_pp_ci95":np.quantile(np.array(differences)[:,1],[.025,.975]).tolist()}
    usage=None
    if (folder/"jev-run.json").exists():
        usage=json.loads((folder/"jev-run.json").read_text())
        usage["successful_input_tokens"]=sum(r["response"]["usage"]["input_tokens"] for r in attempted if r["status"]=="ok")
        usage["successful_output_tokens"]=sum(r["response"]["usage"]["output_tokens"] for r in attempted if r["status"]=="ok")
        if (folder/"jev-invocations.jsonl").exists():
            invocations=read_jsonl(folder/"jev-invocations.jsonl")
            seconds=sum(r["wall_seconds_this_invocation"] for r in invocations)
            usage["all_invocations_wall_seconds"]=seconds
            usage["successful_requests_per_wall_second"]=len(successes)/seconds if seconds else None
    report={"status":"complete" if jev_status["status"]=="complete" else "jev_results_pending",
            "labels":LABELS,"label_names":NAMES,"test_n":len(test),"bootstrap_repeats":repeats,
            "bootstrap_unit":"individual deduplicated narrative; consumer/incident clusters are unavailable",
            "data":data,"local_environment":local,"jev":jev_status,"usage":usage,"methods":metrics,"paired":paired,
            "evidence_sha256":{p.name:digest(p) for p in (folder/"study.jsonl",folder/"requests.jsonl",folder/"local-predictions.jsonl",folder/"jev-responses.jsonl",folder/"jev-attempts.jsonl") if p.exists()},
            "analysis_source_sha256":{p.name:digest(p) for p in sorted(ROOT.glob('*.py'))}}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with (folder/"comparison.csv").open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["method","status","n","accuracy","macro_f1","max_abs_share_error_pp","brier","response_ms_mean","response_ms_p50","response_ms_p95","review_coverage","accepted_error"])
        for method in ("keyword","tfidf","jev"):
            m=metrics.get(method)
            if m is None:
                writer.writerow([method,jev_status["status"]]+[""]*10);continue
            gate=m.get("test_review_point",{})
            writer.writerow([method,"measured",m["n"],m["accuracy"],m["macro_f1"],m["max_abs_share_error_pp"],m.get("brier","not_available"),
                             m["latency_ms"]["mean"],m["latency_ms"]["p50"],m["latency_ms"]["p95"],gate.get("coverage","not_available"),gate.get("accepted_error","not_available")])
    (folder/"review-policy.json").write_text(json.dumps({name:m["review_policy"] for name,m in metrics.items() if "review_policy" in m},indent=2)+"\n")
    errors=[]
    for method,predictions in methods.items():
        for r in test:
            pred=predictions[r["id"]]
            if pred["label"]!=r["label"]:
                errors.append({"id":r["id"],"method":method,"reference_label":r["label"],"prediction":pred["label"],"max_probability":max(pred["probabilities"]) if pred["probabilities"] is not None else None})
    from .study import write_jsonl
    write_jsonl(folder/"errors.jsonl",errors)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=500)
    parser.add_argument("--no-figures",action="store_true")
    args=parser.parse_args()
    if args.bootstrap<100:parser.error("Use at least 100 bootstrap resamples")
    report=build(args.run,args.bootstrap)
    if not args.no_figures:
        from .figures import draw
        draw(args.run,report)
    print(json.dumps({"status":report["status"],"jev":report["jev"],"methods":{name:{k:m[k] for k in ("accuracy","macro_f1","max_abs_share_error_pp")} for name,m in report["methods"].items()}},indent=2))


if __name__=="__main__":main()
