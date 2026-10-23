"""Result charts, always generated from measured result files."""
from pathlib import Path
import os
import tempfile
import textwrap
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"jev-feedback-matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .study import LABELS,NAMES

COLORS={"keyword":"#B56735","tfidf":"#23758C","jev":"#6555A1"}
NICK={"keyword":"Keyword","tfidf":"TF-IDF + LR","jev":"Jev 1.14"}


def draw(folder,report):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold","figure.facecolor":"white",
                         "savefig.facecolor":"white","svg.fonttype":"none"})
    out=Path(folder)/"figures";out.mkdir(exist_ok=True)
    methods=report["methods"]
    available=[m for m in ("keyword","tfidf","jev") if m in methods]
    n=report["test_n"]
    suffix="Jev results pending; no Jev bars or points are fabricated." if "jev" not in methods else "All methods use the same held-out narratives."
    def save(fig,name,footnote):
        wrapped="\n".join(textwrap.wrap(footnote,width=int(fig.get_figwidth()*14)))
        fig.text(.03,.015,wrapped,fontsize=9,color="#445366",va="bottom")
        fig.tight_layout(rect=(0,.15 if name=="confusion-matrices" else .11,1,.95))
        for ext in ("png","svg","pdf"):
            fig.savefig(out/f"{name}.{ext}",dpi=180)
        plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(len(LABELS));width=.8/(len(available)+1)
    truth=np.array(methods["tfidf"]["true_counts"])/n*100
    for i,(label,values,color) in enumerate([("Reference labels",truth,"#293C4A")]+[(NICK[m],np.array(methods[m]["predicted_counts"])/n*100,COLORS[m]) for m in available]):
        ax.bar(x-.4+width/2+i*width,values,width=width,label=label,color=color)
    ax.set_xticks(x,["Bank\naccount","Credit\ncard","Credit\nreporting","Debt\ncollection","Mortgage"])
    ax.set_ylabel("Share of all held-out narratives (%)")
    ax.set_title("Does the classifier preserve the dashboard's product mix?")
    ax.legend(ncol=len(available)+1,loc="upper center",bbox_to_anchor=(.5,1.01),fontsize=10)
    ax.set_ylim(0,max(truth.max(),max(max(methods[m]["predicted_counts"])/n*100 for m in available))*1.25)
    save(fig,"category-shares",f"N={n:,}; one narrative, one primary product label. "+suffix)

    fig,axes=plt.subplots(1,3,figsize=(15,5.4))
    short=["Account","Card","Reporting","Collection","Mortgage"]
    for ax,m in zip(axes,("keyword","tfidf","jev")):
        if m not in methods:
            ax.axis("off");ax.text(.5,.5,"Jev pending\nNo predictions yet",ha="center",va="center",fontsize=16);continue
        matrix=np.array(methods[m]["confusion"])
        denom=matrix.sum(axis=1,keepdims=True)
        rates=np.divide(matrix,denom,out=np.zeros_like(matrix,dtype=float),where=denom!=0)
        ax.imshow(rates,vmin=0,vmax=1,cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f"{matrix[i,j]}\n{100*rates[i,j]:.0f}%",ha="center",va="center",fontsize=8,color="white" if rates[i,j]>.5 else "#203746")
        ax.set_xticks(range(5),short,rotation=45,ha="right")
        ax.set_yticks(range(5),short)
        ax.set_xlabel("Predicted product");ax.set_ylabel("Reference product");ax.set_title(NICK[m])
    fig.suptitle("Which categories donate errors to which other categories?",fontsize=16)
    save(fig,"confusion-matrices",f"N={n:,}; rows normalize to 100%; cells show counts and row percentages. "+suffix)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    x=np.arange(len(available));w=.36
    axes[0].bar(x-w/2,[methods[m]["accuracy"]*100 for m in available],width=w,label="Accuracy",color="#7DA8B8")
    axes[0].bar(x+w/2,[methods[m]["macro_f1"]*100 for m in available],width=w,label="Macro F1",color="#234B62")
    axes[0].set_xticks(x,[NICK[m] for m in available]);axes[0].set_ylim(0,100);axes[0].set_ylabel("Score (%)");axes[0].legend()
    axes[0].set_title("Individual classification")
    for i,m in enumerate(available):
        value=methods[m]["max_abs_share_error_pp"]
        axes[1].bar(i,value,color=COLORS[m]);axes[1].text(i,value+.07,f"{value:.2f}",ha="center")
    axes[1].set_xticks(x,[NICK[m] for m in available]);axes[1].set_ylabel("Largest absolute share error (percentage points)")
    axes[1].set_title("Dashboard distortion")
    save(fig,"metrics-overview",f"Same N={n:,}. Full bootstrap intervals are in metrics.json. "+suffix)

    fig,ax=plt.subplots(figsize=(10,5.5))
    for m in available:
        if "review_curve" not in methods[m]:continue
        points=[p for p in methods[m]["review_curve"] if p["accepted_error"] is not None]
        ax.plot([100*p["coverage"] for p in points],[100*p["accepted_error"] for p in points],label=NICK[m],color=COLORS[m],lw=2)
        selected=methods[m]["test_review_point"]
        if selected["accepted_error"] is not None:
            ax.scatter([100*selected["coverage"]],[100*selected["accepted_error"]],marker="*",s=200,color=COLORS[m],edgecolor="#1F2937",zorder=4)
    ax.axhline(10,color="#A75036",ls="--",lw=1,label="Development selection target (10%)")
    ax.set_xlabel("Automatically accepted share of held-out messages (%)")
    ax.set_ylabel("Error among accepted messages (%)")
    ax.set_title("Review less—or accept more classification errors?")
    ax.set_xlim(0,102);ax.set_ylim(bottom=0);ax.legend(fontsize=10)
    save(fig,"review-tradeoff","Stars mark thresholds selected on development data, then evaluated on test. Curves are descriptive. "+suffix)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    probabilistic=[m for m in available if "brier" in methods[m]]
    axes[0].plot([0,1],[0,1],color="#8B9299",ls="--",label="Ideal top-label calibration")
    for m in probabilistic:
        bins=[b for b in methods[m]["reliability"] if b["n"]]
        axes[0].plot([b["mean_top_probability"] for b in bins],[b["accuracy"] for b in bins],marker="o",color=COLORS[m],label=NICK[m])
        for b in bins:
            axes[0].annotate(str(b["n"]),(b["mean_top_probability"],b["accuracy"]),xytext=(3,5),textcoords="offset points",fontsize=8)
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel="Mean maximum probability in bin",ylabel="Observed accuracy",title="Reliability, with bin counts")
    axes[0].legend(fontsize=9)
    for i,m in enumerate(probabilistic):
        v=methods[m]["brier"];axes[1].bar(i,v,color=COLORS[m]);axes[1].text(i,v+.004,f"{v:.3f}",ha="center")
    axes[1].set_xticks(range(len(probabilistic)),[NICK[m] for m in probabilistic]);axes[1].set_ylabel("Multiclass Brier score (sum over classes)")
    axes[1].set_title("Probability error: lower is better");axes[1].set_ylim(bottom=0)
    save(fig,"probability-quality","Keyword rules provide no probabilities, so Brier/calibration are not assigned to them. "+suffix)

    fig,ax=plt.subplots(figsize=(10,4.6))
    for i,m in enumerate(available):
        timing=methods[m]["latency_ms"]
        ax.plot([timing["p50"],timing["p95"]],[i,i],color=COLORS[m],lw=3)
        ax.scatter([timing["p50"]],[i],s=70,color=COLORS[m],label="p50" if i==0 else None)
        ax.scatter([timing["p95"]],[i],s=70,marker="D",facecolors="white",edgecolors=COLORS[m],label="p95" if i==0 else None)
        ax.scatter([timing["mean"]],[i],s=70,marker="x",color=COLORS[m],label="mean" if i==0 else None)
        ax.annotate(f'{timing["p50"]:.3f} / {timing["mean"]:.3f} / {timing["p95"]:.3f}',(timing["p95"],i),xytext=(8,9),textcoords="offset points",fontsize=9)
    ax.set_xscale("log");ax.set_yticks(range(len(available)),[NICK[m] for m in available]);ax.set_ylim(-.5,len(available)-.3)
    ax.set_xlabel("Successful prediction/HTTP response time (ms, logarithmic scale)")
    ax.set_title("Observed response time in this implementation");ax.legend(loc="lower right")
    ax.margins(x=.3)
    save(fig,"latency","Labels: p50 / mean / p95. Local warm CPU calls versus full HTTP round trips; these are not equivalent compute measurements. "+suffix)
