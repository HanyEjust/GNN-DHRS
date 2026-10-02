import sys,numpy as np,pandas as pd,torch
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold,train_test_split
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.ablations import make_variant
from gnn_dhrs.train import class_weights,weighted_bce
from gnn_dhrs.metrics import evaluate,select_threshold
from gnn_dhrs.repro import set_seed
cfg=load_config();rows=[];variants=['no_graph_mlp','uniform_aggregation','attention_no_similarity','full_gnn_dhrs']
def fit(model,X,y,ei,ew,epochs=120):
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=model.to(dev);X=torch.tensor(X,dtype=torch.float32,device=dev);y=torch.tensor(y,dtype=torch.long,device=dev);ei=ei.to(dev);ew=ew.to(dev);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005);w=class_weights(y.cpu().numpy()).to(dev)
 for _ in range(epochs):model.train();opt.zero_grad();z=model(X,ei,ew);loss=weighted_bce(z,y,w);loss.backward();opt.step()
 return model
def prob(model,X,ei,ew):
 dev=next(model.parameters()).device;model.eval();
 with torch.no_grad():return torch.sigmoid(model(torch.tensor(X,dtype=torch.float32,device=dev),ei.to(dev),ew.to(dev))).cpu().numpy()
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);A,T=pp.transform(ds.X.iloc[tr]),pp.transform(ds.X.iloc[te]);D,y,_,_=dhrs(A,ds.y[tr],11,1.0,pp.onehot_groups);ei,ew=directed_cosine_knn(D,10)
  for v in variants:
   set_seed(11); m=make_variant(v,D.shape[1],cfg); usew=torch.ones_like(ew) if v=='attention_no_similarity' else ew; m=fit(m,D,y,ei,usew); # evaluate test nodes jointly but no test-test edges
   from sklearn.metrics.pairwise import cosine_similarity
   ps=[]
   for q in T:
    s=cosine_similarity(q.reshape(1,-1),D)[0];nbr=np.where(s>0)[0];nbr=nbr[np.argsort(-s[nbr])[:10]];X=np.vstack([D,q]);qe=torch.tensor([nbr,[len(D)]*len(nbr)],dtype=torch.long);qw=torch.tensor(s[nbr],dtype=torch.float32);E=torch.cat([ei,qe],1);W=torch.cat([usew,torch.ones_like(qw) if v=='attention_no_similarity' else qw]);ps.append(prob(m,X,E,W)[-1])
   rows.append({'dataset':dn,'fold':fold,'variant':v,**evaluate(ds.y[te],np.array(ps),.5)})
pd.DataFrame(rows).to_csv('outputs/tables/graph_component_ablation.csv',index=False)
