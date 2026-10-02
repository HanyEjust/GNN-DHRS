import numpy as np, torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def directed_cosine_knn(X,k=10,positive_only=True):
    X=np.asarray(X,float); S=cosine_similarity(X); np.fill_diagonal(S,-np.inf); src=[];dst=[];w=[]
    for target in range(len(X)):
        cand=np.where(S[target]>0)[0] if positive_only else np.arange(len(X))
        cand=cand[cand!=target]; nbr=cand[np.argsort(-S[target,cand])[:k]]
        # PyG edge_index source -> target; manuscript i(target) receives from j(neighbor)
        src.extend(nbr.tolist()); dst.extend([target]*len(nbr)); w.extend(S[target,nbr].tolist())
    return torch.tensor([src,dst],dtype=torch.long),torch.tensor(w,dtype=torch.float32)

def inductive_edges(X_train_real,xq,k=10):
    s=cosine_similarity(np.asarray(xq).reshape(1,-1),np.asarray(X_train_real))[0]; cand=np.where(s>0)[0]; nbr=cand[np.argsort(-s[cand])[:k]]
    # local graph indices: training 0..n-1, query n
    src=nbr; dst=np.full(len(nbr),len(X_train_real)); return torch.tensor([src,dst],dtype=torch.long),torch.tensor(s[nbr],dtype=torch.float32)

def topology_stats(edge_index,y,synthetic=None):
    s=edge_index[0].cpu().numpy(); d=edge_index[1].cpu().numpy(); n=len(y); E=len(s)
    hom=float(np.mean(np.asarray(y)[s]==np.asarray(y)[d])) if E else np.nan
    out=np.bincount(s,minlength=n); inn=np.bincount(d,minlength=n)
    r={"nodes":n,"edges":E,"mean_in_degree":float(inn.mean()),"mean_out_degree":float(out.mean()),"homophily":hom}
    if synthetic is not None:
        syn=np.asarray(synthetic,bool); r.update(real_real=int(np.sum(~syn[s]&~syn[d])),real_synthetic=int(np.sum(syn[s]^syn[d])),synthetic_synthetic=int(np.sum(syn[s]&syn[d])))
    return r
