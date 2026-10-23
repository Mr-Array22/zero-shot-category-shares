#!/usr/bin/env python3
"""Fit the local text baseline; measure individual, warm-model predictions."""
import argparse
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .study import LABELS, digest, keyword, object_hash, read_jsonl, write_jsonl


def run(folder):
    manifest=json.loads((folder/"data-manifest.json").read_text())
    if digest(folder/"study.jsonl") != manifest["study_sha256"]:
        raise ValueError("Study data hash changed")
    records=read_jsonl(folder/"study.jsonl")
    train=[r for r in records if r["split"]=="train"]
    evaluation=[r for r in records if r["split"]!="train"]
    if set(r["label"] for r in train)!=set(LABELS):
        raise ValueError("Training set must contain every category")
    pipeline=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=40000,sublinear_tf=True)),
                       ("classifier",LogisticRegression(C=1.0,solver="lbfgs",max_iter=1500,random_state=20261022))])
    start=time.perf_counter()
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        pipeline.fit([r["text"] for r in train],[r["label"] for r in train])
    fit_seconds=time.perf_counter()-start
    indices=[list(pipeline.classes_).index(label) for label in LABELS]
    outputs=[]
    timings={}
    for method in ("keyword","tfidf"):
        started=time.perf_counter()
        with threadpool_limits(limits=1):
            for row in evaluation:
                tick=time.perf_counter()
                if method=="keyword":
                    label=keyword(row["text"])
                    probs=None
                else:
                    probs=pipeline.predict_proba([row["text"]])[0][indices].tolist()
                    label=LABELS[int(np.argmax(probs))]
                latency=(time.perf_counter()-tick)*1000
                outputs.append({"id":row["id"],"split":row["split"],"method":method,
                                "label":label,"probabilities":probs,"latency_ms":latency,
                                "input_hash":object_hash(row["text"])})
        timings[method]={"evaluation_wall_seconds":time.perf_counter()-started,"requests":len(evaluation)}
    write_jsonl(folder/"local-predictions.jsonl",outputs)
    vectorizer=pipeline.named_steps["tfidf"]
    model=pipeline.named_steps["classifier"]
    (folder/"local-manifest.json").write_text(json.dumps({
        "study_sha256":digest(folder/"study.jsonl"),"predictions_sha256":digest(folder/"local-predictions.jsonl"),
        "python":platform.python_version(),"platform":platform.platform(),"sklearn":sklearn.__version__,
        "numpy":np.__version__,"threadpool_limit":1,"fit_seconds":fit_seconds,
        "vocabulary_features":len(vectorizer.vocabulary_),"iterations":model.n_iter_.tolist(),
        "parameters":{"C":1.0,"max_iter":1500,"solver":"lbfgs","ngram_range":[1,2],"min_df":3,"max_features":40000,"sublinear_tf":True},
        "timing_design":"Single records; model remains in memory; no batching; fit excluded. Local CPU timing is not hosted-model compute latency.",
        "timings":timings,"model_calls":0,"source_sha256":{p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    },indent=2)+"\n")
    print(json.dumps({"fit_seconds":round(fit_seconds,3),"features":len(vectorizer.vocabulary_),"evaluation_records":len(evaluation),"saved":str(folder/"local-predictions.jsonl")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    run(parser.parse_args().run)
