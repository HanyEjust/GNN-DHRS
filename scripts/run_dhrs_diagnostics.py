import sys,pandas as pd,numpy as np
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
cfg=load_config();rows=[]
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);A=pp.transform(ds.X.iloc[tr]);D,y,syn,a=dhrs(A,ds.y[tr],11,1.0,pp.onehot_groups);rows.append({'dataset':dn,'fold':fold,'stage':'before','stroke':int(ds.y[tr].sum()),'nonstroke':int((1-ds.y[tr]).sum()),'total':len(tr)});rows.append({'dataset':dn,'fold':fold,'stage':'after_borderline','total':a['n_after_borderline']});rows.append({'dataset':dn,'fold':fold,'stage':'after_dhrs','stroke':a['stroke_after'],'nonstroke':a['nonstroke_after'],'total':a['n_after_dhrs'],'synthetic':int(syn.sum())})
pd.DataFrame(rows).to_csv('outputs/tables/dhrs_stage_distribution.csv',index=False)
