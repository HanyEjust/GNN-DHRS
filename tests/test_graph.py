import sys,numpy as np
sys.path.insert(0,'src')
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.graph_variants import graph_from_definition
def test_knn_has_no_self_edges():
 X=np.eye(12)+.01;ei,ew=directed_cosine_knn(X,3);assert not np.any(ei[0].numpy()==ei[1].numpy())
def test_graph_variants_construct():
 X=np.random.default_rng(1).normal(size=(20,5))
 for g in ['cosine_knn','euclidean_knn','gower_knn','mutual_cosine_knn','cosine_threshold']:
  ei,ew=graph_from_definition(X,g,3);assert ei.shape[0]==2;assert len(ew)==ei.shape[1]
