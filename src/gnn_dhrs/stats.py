import numpy as np
from scipy.stats import t

def nadeau_bengio_corrected(d,test_train_ratio=1/9,alpha=.05):
 d=np.asarray(d,float);n=len(d);mu=d.mean();sd=d.std(ddof=1);se=np.sqrt((1/n+test_train_ratio)*sd**2);df=max(1,n-1);crit=t.ppf(1-alpha/2,df);stat=mu/se if se else np.inf;p=2*t.sf(abs(stat),df);dz=mu/sd if sd else np.inf;return {'mean_difference':mu,'ci_low':mu-crit*se,'ci_high':mu+crit*se,'dz':dz,'p':p}
def holm(p):
 p=np.asarray(p,float);m=len(p);order=np.argsort(p);adj=np.empty(m);running=0
 for rank,i in enumerate(order):running=max(running,(m-rank)*p[i]);adj[i]=min(1,running)
 return adj
