import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,matthews_corrcoef,balanced_accuracy_score,cohen_kappa_score,brier_score_loss,confusion_matrix

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); out=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
        if m.any(): out+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)
def select_threshold(y,p,candidates):
    scores=[f1_score(y,np.asarray(p)>=t,zero_division=0) for t in candidates]; return float(candidates[int(np.argmax(scores))])
def evaluate(y,p,t=.5,ece_bins=10):
    y=np.asarray(y,int); p=np.asarray(p,float); yh=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    spec=tn/(tn+fp) if tn+fp else np.nan; npv=tn/(tn+fn) if tn+fn else np.nan
    safe=lambda fn: float(fn(y,p)) if len(np.unique(y))>1 else np.nan
    return {"accuracy":accuracy_score(y,yh),"precision":precision_score(y,yh,zero_division=0),"recall":recall_score(y,yh,zero_division=0),"specificity":spec,"npv":npv,"f1":f1_score(y,yh,zero_division=0),"balanced_accuracy":balanced_accuracy_score(y,yh),"mcc":matthews_corrcoef(y,yh),"kappa":cohen_kappa_score(y,yh),"roc_auc":safe(roc_auc_score),"pr_auc":safe(average_precision_score),"brier":brier_score_loss(y,p),"ece":ece(y,p,ece_bins),"tp":tp,"tn":tn,"fp":fp,"fn":fn}
def selective_curve(y,p,coverages):
    conf=np.maximum(p,1-p); order=np.argsort(-conf); rows=[]
    for c in coverages:
        n=max(1,int(np.ceil(c*len(y)))); idx=order[:n]; yh=(p[idx]>=.5).astype(int); rows.append({"coverage":n/len(y),"selective_risk":float(np.mean(yh!=y[idx])),"accuracy":accuracy_score(y[idx],yh)})
    return rows
