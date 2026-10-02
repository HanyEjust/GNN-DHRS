"""Sensitivity experiment: projected one-hot DHRS versus categorical-aware SMOTENC followed by ENN.
SMOTENC is fitted only on each outer-training partition. Categorical columns are ordinal-encoded for SMOTENC, then the resampled rows are decoded and passed through the same fold-specific model preprocessor.
"""
import sys,numpy as np,pandas as pd,torch
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics.pairwise import cosine_similarity
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.train import train_fixed_epochs,predict
from gnn_dhrs.metrics import evaluate
cfg=load_config();rows=[]
def smotenc_enn_raw(X,y,seed=11):
 X=X.copy();num=[c for c in X if pd.api.types.is_numeric_dtype(X[c])];cat=[c for c in X if c not in num]; ni=SimpleImputer(strategy='mean').fit(X[num]);ci=SimpleImputer(strategy='most_frequent').fit(X[cat]);enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1).fit(ci.transform(X[cat]));Z=np.c_[ni.transform(X[num]),enc.transform(ci.transform(X[cat]))];catidx=list(range(len(num),Z.shape[1]));Z,y2=SMOTENC(categorical_features=catidx,sampling_strategy=1.0,random_state=seed,k_neighbors=5).fit_resample(Z,y);Z,y2=EditedNearestNeighbours(n_neighbors=3,sampling_strategy='all').fit_resample(Z,y2);out=pd.DataFrame(index=np.arange(len(Z)));out[num]=Z[:,:len(num)];cats=enc.inverse_transform(np.rint(Z[:,len(num):]).clip(0,[len(c)-1 for c in enc.categories_]));out[cat]=cats;return out,np.asarray(y2)
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);A,T=pp.transform(ds.X.iloc[tr]),pp.transform(ds.X.iloc[te]);D,y,_,_=dhrs(A,ds.y[tr],11,1.0,pp.onehot_groups); raw,yraw=smotenc_enn_raw(ds.X.iloc[tr],ds.y[tr],11);M=pp.transform(raw)
  for mode,X,Y in [('projected_onehot_dhrs',D,y),('smotenc_enn',M,yraw)]:
   ei,ew=directed_cosine_knn(X,10);m=train_fixed_epochs(X,Y,ei,ew,cfg,11,120);ps=[]
   for q in T:
    s=cosine_similarity(q.reshape(1,-1),X)[0];nbr=np.where(s>0)[0];nbr=nbr[np.argsort(-s[nbr])[:10]];qe=torch.tensor([nbr,[len(X)]*len(nbr)],dtype=torch.long);qw=torch.tensor(s[nbr],dtype=torch.float32);ps.append(predict(m,np.vstack([X,q]),torch.cat([ei,qe],1),torch.cat([ew,qw]))[-1])
   rows.append({'dataset':dn,'fold':fold,'treatment':mode,**evaluate(ds.y[te],np.array(ps),.5)})
pd.DataFrame(rows).to_csv('outputs/tables/categorical_resampling_sensitivity.csv',index=False)
