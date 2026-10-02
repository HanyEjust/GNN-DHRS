import numpy as np, torch
from .train import predict

def attention_edges(model,X,ei,ew):
    p,(e,a)=predict(model,X,ei,ew,attention=True); score=a.mean(1).numpy(); src=e[0].numpy(); dst=e[1].numpy(); return p,src,dst,score

def remove_edge_indices(ei,ew,indices):
    keep=np.ones(ei.shape[1],bool); keep[np.asarray(indices,int)]=False; return ei[:,keep],ew[keep]

def top_edge_ablation(model,X,ei,ew,budget=5):
    p,s,d,a=attention_edges(model,X,ei,ew); nonself=np.where(s!=d)[0]; top=nonself[np.argsort(-a[nonself])[:budget]]; e2,w2=remove_edge_indices(ei,ew,top); return p,predict(model,X,e2,w2),top

def random_edge_ablation(model,X,ei,ew,budget=5,repeats=10,seed=11):
    rng=np.random.default_rng(seed); s=ei[0].numpy();d=ei[1].numpy(); cand=np.where(s!=d)[0]; outs=[]
    for _ in range(repeats):
        idx=rng.choice(cand,min(budget,len(cand)),False); e2,w2=remove_edge_indices(ei,ew,idx); outs.append(predict(model,X,e2,w2))
    return np.asarray(outs)
