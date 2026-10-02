from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt

def class_distribution(datasets,out):
    names=[];ns=[];ss=[]
    for name,ds in datasets.items(): names.append(name);ss.append(int(ds.y.sum()));ns.append(int((1-ds.y).sum()))
    x=np.arange(len(names));fig,ax=plt.subplots(figsize=(8,5));ax.bar(x,ns,label="Non-stroke");ax.bar(x,ss,bottom=ns,label="Stroke");ax.set_xticks(x,names);ax.set_ylabel("Patients");ax.legend();fig.tight_layout();Path(out).parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=300);plt.close(fig)
def roc_pr_curves(preds,out_prefix):
    from sklearn.metrics import roc_curve,precision_recall_curve,auc
    for kind in ["roc","pr"]:
        fig,ax=plt.subplots(figsize=(6,5))
        for name,g in preds.groupby("dataset"):
            if kind=="roc": x,y,_=roc_curve(g.y,g.p); lab=f"{name} AUC={auc(x,y):.3f}";ax.plot(x,y,label=lab);ax.set(xlabel="False Positive Rate",ylabel="True Positive Rate")
            else: y,x,_=precision_recall_curve(g.y,g.p); lab=f"{name} AP={auc(x,y):.3f}";ax.plot(x,y,label=lab);ax.set(xlabel="Recall",ylabel="Precision")
        ax.legend();fig.tight_layout();fig.savefig(f"{out_prefix}_{kind}.png",dpi=300);plt.close(fig)
def calibration_plot(preds,out,bins=10):
    from sklearn.calibration import calibration_curve
    fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],"--")
    for n,g in preds.groupby("dataset"):
        y,x=calibration_curve(g.y,g.p,n_bins=bins,strategy="uniform");ax.plot(x,y,marker="o",label=n)
    ax.set(xlabel="Mean predicted probability",ylabel="Observed frequency");ax.legend();fig.tight_layout();fig.savefig(out,dpi=300);plt.close(fig)

def grouped_line(df,x,y,group,out,title=None):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for name,g in df.groupby(group): ax.plot(g[x],g[y],marker='o',label=str(name))
    ax.set_xlabel(x.replace('_',' ').title());ax.set_ylabel(y.replace('_',' ').title());
    if title:ax.set_title(title)
    ax.legend();fig.tight_layout();fig.savefig(out,dpi=220);plt.close(fig)
