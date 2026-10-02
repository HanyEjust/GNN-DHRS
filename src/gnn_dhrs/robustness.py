import numpy as np

def add_missingness(X,rate,seed=11):
    rng=np.random.default_rng(seed); Z=np.asarray(X,float).copy(); mask=rng.random(Z.shape)<rate; Z[mask]=np.nan
    # test-time imputation in standardized space: training mean is zero
    return np.nan_to_num(Z,nan=0.0)
def add_gaussian_noise(X,sigma,seed=11): return np.asarray(X)+np.random.default_rng(seed).normal(0,sigma,np.asarray(X).shape)
def prevalence_subsample(y,multiplier,seed=11):
    y=np.asarray(y);rng=np.random.default_rng(seed); pos=np.where(y==1)[0];neg=np.where(y==0)[0]; base=len(pos)/len(y); target=np.clip(base*multiplier,1e-6,.999999)
    if target<=base:
        npos=min(len(pos),int(round(target*len(neg)/(1-target)))); return np.sort(np.r_[rng.choice(pos,npos,False),neg])
    nneg=min(len(neg),int(round(len(pos)*(1-target)/target))); return np.sort(np.r_[pos,rng.choice(neg,nneg,False)])
def corrupt_attention_neighbors(neighbors,rate,reference_n,seed=11):
    rng=np.random.default_rng(seed); n=len(neighbors); k=int(round(rate*n)); out=np.array(neighbors,copy=True)
    if k: out[rng.choice(n,k,False)]=rng.integers(0,reference_n,k)
    return out
