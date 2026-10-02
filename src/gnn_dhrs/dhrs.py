import numpy as np
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, SMOTENC
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

def project_onehot(X,groups):
    X=np.asarray(X,float).copy()
    for g in groups:
        if not g: continue
        winner=np.argmax(X[:,g],axis=1); X[:,g]=0.0; X[np.arange(len(X)),np.asarray(g)[winner]]=1.0
    return X

def dhrs(X,y,seed=11,ratio=1.0,onehot_groups=None,k=5,m=10,enn_k=3):
    before=len(y); sm1=BorderlineSMOTE(sampling_strategy=ratio,random_state=seed,k_neighbors=k,m_neighbors=m,kind="borderline-1")
    X1,y1=sm1.fit_resample(X,y); X1=project_onehot(X1,onehot_groups or [])
    sm=SMOTE(sampling_strategy="auto",random_state=seed,k_neighbors=k)
    enn=EditedNearestNeighbours(n_neighbors=enn_k,sampling_strategy="all")
    X2,y2=SMOTEENN(smote=sm,enn=enn,random_state=seed).fit_resample(X1,y1); X2=project_onehot(X2,onehot_groups or [])
    synthetic=np.zeros(len(y2),dtype=bool)
    # SMOTEENN does not preserve provenance; conservative provenance mask based on rows absent from original X
    originals={np.asarray(r,dtype=np.float64).tobytes() for r in np.asarray(X)}
    synthetic=np.array([np.asarray(r,dtype=np.float64).tobytes() not in originals for r in X2])
    audit={"n_before":before,"n_after_borderline":len(y1),"n_after_dhrs":len(y2),"stroke_after":int(np.sum(y2==1)),"nonstroke_after":int(np.sum(y2==0))}
    return X2,np.asarray(y2),synthetic,audit

def resample_variant(name,X,y,seed=11,ratio=1.0,groups=None):
    if name=="none": return np.asarray(X),np.asarray(y)
    if name=="smote": Z,t=SMOTE(sampling_strategy=ratio,random_state=seed).fit_resample(X,y)
    elif name=="borderline": Z,t=BorderlineSMOTE(sampling_strategy=ratio,random_state=seed).fit_resample(X,y)
    elif name=="smoteenn": Z,t=SMOTEENN(random_state=seed).fit_resample(X,y)
    elif name=="bsm_enn":
        Z,t=BorderlineSMOTE(sampling_strategy=ratio,random_state=seed).fit_resample(X,y); Z,t=EditedNearestNeighbours(n_neighbors=3,sampling_strategy="all").fit_resample(Z,t)
    elif name=="reverse_dhrs":
        Z,t=SMOTEENN(random_state=seed).fit_resample(X,y); Z,t=BorderlineSMOTE(sampling_strategy=ratio,random_state=seed).fit_resample(Z,t)
    else: raise ValueError(name)
    return project_onehot(Z,groups or []),np.asarray(t)
