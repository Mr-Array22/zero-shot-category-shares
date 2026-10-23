"""Independent classification, aggregate-bias, probability and review metrics."""
import numpy as np
from .study import LABELS


def confusion(truth,pred):
    return np.bincount(np.asarray(truth)*len(LABELS)+np.asarray(pred),minlength=len(LABELS)**2).reshape(len(LABELS),len(LABELS))


def scalar_metrics(truth,pred,probabilities=None):
    matrix=confusion(truth,pred)
    total=matrix.sum()
    tp=np.diag(matrix)
    denom=matrix.sum(axis=0)+matrix.sum(axis=1)
    f1=np.divide(2*tp,denom,out=np.zeros(len(LABELS),dtype=float),where=denom!=0)
    delta=(matrix.sum(axis=0)-matrix.sum(axis=1))/total
    result={"n":int(total),"accuracy":float(tp.sum()/total),"macro_f1":float(f1.mean()),
            "confusion":matrix.tolist(),"per_class_f1":f1.tolist(),"true_counts":matrix.sum(axis=1).tolist(),
            "predicted_counts":matrix.sum(axis=0).tolist(),"share_error_pp":(delta*100).tolist(),
            "max_abs_share_error_pp":float(max(abs(delta))*100),"total_variation":float(abs(delta).sum()/2)}
    if probabilities is not None:
        ps=np.asarray(probabilities,dtype=float)
        if ps.shape!=(len(truth),len(LABELS)) or not np.isfinite(ps).all() or (ps<0).any() or (ps>1).any() or not np.allclose(ps.sum(axis=1),1,atol=1e-5):
            raise ValueError("Invalid probability matrix")
        onehot=np.eye(len(LABELS))[truth]
        result["brier"] = float(np.mean(np.sum((ps-onehot)**2,axis=1)))
        result["soft_share_error_pp"] = ((ps.mean(axis=0)-onehot.mean(axis=0))*100).tolist()
    return result


def bootstrap(truth,pred,probabilities=None,repeats=500,seed=20261022):
    # Same seed and test IDs across methods gives paired resamples.
    rng=np.random.default_rng(seed)
    values={name:[] for name in ("accuracy","macro_f1","max_abs_share_error_pp","total_variation")}
    if probabilities is not None:
        values["brier"]=[]
    for _ in range(repeats):
        ix=rng.integers(0,len(truth),len(truth))
        m=scalar_metrics(truth[ix],pred[ix],probabilities[ix] if probabilities is not None else None)
        for name in values:
            values[name].append(m[name])
    return {name:np.quantile(v,[.025,.975]).tolist() for name,v in values.items()}


def review_point(truth,pred,probabilities,threshold):
    accepted=np.max(probabilities,axis=1)>=threshold
    n=int(accepted.sum())
    review=~accepted
    result={"threshold":float(threshold),"accepted":n,"reviewed":int(review.sum()),"coverage":float(accepted.mean()),
            "accepted_error":float(np.mean(pred[accepted]!=truth[accepted])) if n else None,
            "review_counts_by_true_class":np.bincount(truth[review],minlength=len(LABELS)).tolist(),
            "accepted_predicted_counts":np.bincount(pred[accepted],minlength=len(LABELS)).tolist()}
    # Counterfactual: perfect review replaces uncertain predictions with reference
    # labels. This is an explicitly idealized bound, not measured human accuracy.
    hybrid=pred.copy()
    hybrid[review]=truth[review]
    result["oracle_review_max_abs_share_error_pp"]=scalar_metrics(truth,hybrid)["max_abs_share_error_pp"]
    return result


def choose_threshold(truth,pred,probabilities,target=.10,minimum=100):
    options=[]
    for threshold in np.linspace(0,1,101):
        point=review_point(truth,pred,probabilities,float(threshold))
        if point["accepted"]>=minimum and point["accepted_error"]<=target:
            options.append(point)
    if not options:
        return {"threshold":1.000001,"reason":"No development threshold met the observed-error target with minimum support; review all.","target_error":target,"min_accepted":minimum}
    selected=max(options,key=lambda p:(p["coverage"],-p["threshold"]))
    return {"threshold":selected["threshold"],"reason":"Maximum development coverage meeting observed accepted-error target on fixed grid.","target_error":target,"min_accepted":minimum,"dev_point":selected}


def reliability(truth,pred,probabilities,bins=10):
    top=probabilities.max(axis=1)
    result=[]
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins
        mask=(top>=lo)&((top<=hi) if i==bins-1 else (top<hi))
        result.append({"low":lo,"high":hi,"n":int(mask.sum()),
                       "mean_top_probability":float(top[mask].mean()) if mask.any() else None,
                       "accuracy":float((truth[mask]==pred[mask]).mean()) if mask.any() else None})
    return result
