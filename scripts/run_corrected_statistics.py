import sys,pandas as pd
sys.path.insert(0,'src')
from gnn_dhrs.stats import nadeau_bengio_corrected,holm
m=pd.read_csv('outputs/tables/nested_cv_fold_metrics.csv');b=pd.read_csv('outputs/tables/graph_baselines.csv');rows=[]
for d in sorted(set(m.dataset)&set(b.dataset)):
 for metric in ['recall','f1','pr_auc','mcc']:
  a=m[m.dataset==d][['seed','fold',metric]].rename(columns={metric:'a'});g=b[(b.dataset==d)&(b.model=='GAT-CBF')][['seed','fold',metric]].rename(columns={metric:'b'});q=a.merge(g,on=['seed','fold']);
  if len(q):rows.append({'dataset':d,'metric':metric,**nadeau_bengio_corrected((q.a-q.b).to_numpy(),1/9)})
out=pd.DataFrame(rows)
if len(out):out['p_holm']=holm(out.p.to_numpy())
out.to_csv('outputs/tables/corrected_statistical_validation.csv',index=False)
