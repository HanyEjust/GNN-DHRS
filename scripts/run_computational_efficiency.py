import sys,numpy as np,pandas as pd,time,torch
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.train import train_fixed_epochs,predict
from gnn_dhrs.profiling import profile_call,parameter_count
cfg=load_config();rows=[]
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);A,T=pp.transform(ds.X.iloc[tr]),pp.transform(ds.X.iloc[te]);D,y,_,_=dhrs(A,ds.y[tr],11,1.0,pp.onehot_groups);(ge,gt)=profile_call(directed_cosine_knn,D,10);ei,ew=ge;(m,pt)=profile_call(train_fixed_epochs,D,y,ei,ew,cfg,11,120);t0=time.perf_counter();_ = predict(m,D,ei,ew);infer=(time.perf_counter()-t0)/len(D)*1000;rows.append({'dataset':dn,'fold':fold,'nodes':len(D),'edges':ei.shape[1],'parameters':parameter_count(m),'graph_seconds':gt['seconds'],'train_seconds':pt['seconds'],'cpu_peak_mb':pt['cpu_peak_mb'],'gpu_peak_mb':pt['gpu_peak_mb'],'inference_ms_per_patient_proxy':infer})
pd.DataFrame(rows).to_csv('outputs/tables/computational_efficiency.csv',index=False)
