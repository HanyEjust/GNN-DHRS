import sys,numpy as np,pandas as pd
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.train import train_fixed_epochs,predict
from gnn_dhrs.metrics import evaluate
cfg=load_config();rows=[]
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);A,T=pp.transform(ds.X.iloc[tr]),pp.transform(ds.X.iloc[te]);D,y,_,_=dhrs(A,ds.y[tr],11,1.0,pp.onehot_groups);ei,ew=directed_cosine_knn(D,10)
  for loss in ['wbce','bce']:
   m=train_fixed_epochs(D,y,ei,ew,cfg,11,120,loss); from sklearn.metrics.pairwise import cosine_similarity; import torch;ps=[]
   for q in T:
    s=cosine_similarity(q.reshape(1,-1),D)[0];nbr=np.where(s>0)[0];nbr=nbr[np.argsort(-s[nbr])[:10]];qe=torch.tensor([nbr,[len(D)]*len(nbr)],dtype=torch.long);qw=torch.tensor(s[nbr],dtype=torch.float32);ps.append(predict(m,np.vstack([D,q]),torch.cat([ei,qe],1),torch.cat([ew,qw]))[-1])
   rows.append({'dataset':dn,'fold':fold,'loss':loss,**evaluate(ds.y[te],np.array(ps),.5)})
pd.DataFrame(rows).to_csv('outputs/tables/class_weight_ablation.csv',index=False)
