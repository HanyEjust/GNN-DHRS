import sys,numpy as np
sys.path.insert(0,'src')
from gnn_dhrs.dhrs import dhrs
def test_dhrs_returns_aligned_arrays():
 rng=np.random.default_rng(2);X=rng.normal(size=(100,6));y=np.r_[np.zeros(80,int),np.ones(20,int)];Z,t,s,a=dhrs(X,y,11,1.0,[],3,5,3);assert len(Z)==len(t)==len(s);assert a['n_before']==100
