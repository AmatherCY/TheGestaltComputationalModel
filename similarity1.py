import search_funcs as s
import matplotlib.pyplot as plt
import numpy as np  
from scipy.spatial import distance
from scipy.sparse import csr_matrix  
from scipy.sparse.csgraph import depth_first_order  
from scipy.sparse.csgraph import connected_components  
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix  
from collections import deque
import networkx as nx
import ripser
import warnings  
warnings.filterwarnings("ignore") 

max_dis=500

xr=[]
yr=[]
zr=[]
xb=[]
yb=[]
zb=[]
bz=0
rz=0
e=0.0
red_x=list(np.random.uniform(-e, e, 3)+np.linspace(5,25,3))
blue_x=list(np.random.uniform(-e, e, 4)+np.linspace(0,30,4))
_y=list(np.linspace(0,25,6))
for i in blue_x:
    for j in _y:
        xb.append(i)
        yb.append(j)
        zb.append(bz)

for i in red_x:
    for j in _y:
        xr.append(i)
        yr.append(j)
        zr.append(rz)
x=np.append(xr,xb)
y=np.append(yr,yb)
z=np.append(zr,zb)

pr=np.array(list(zip(xr, yr, zr)),dtype=np.float32)
pb=np.array(list(zip(xb, yb, zb)),dtype=np.float32)
p = np.array(list(zip(x, y, z)),dtype=np.float32)

plt.scatter(xr, yr,color='r')
plt.scatter(xb, yb,color='r')
plt.axis('equal')
plt.show()


import gudhi as gd
rips = gd.RipsComplex(points=np.array(p), max_edge_length = max_dis)
simplex_tree = rips.create_simplex_tree(max_dimension=2)
diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
gd.plot_persistence_diagram(diag,legend=True)
plt.show()
dgm_data=ripser.ripser(p)['dgms']

# Treating infinity points by borrowing from GUDHI's method of plotting PDs
inf_delta=0.1
if len(dgm_data[1])>0:
    max_death=max(np.array(sorted(dgm_data[0], key=s.custom_compare, reverse=True))[1][1],np.array(sorted(dgm_data[1], key=s.custom_compare, reverse=True))[0][1])
else:
    max_death=np.array(sorted(dgm_data[0], key=s.custom_compare, reverse=True))[1][1]
min_birth=0
delta = (max_death - min_birth) * inf_delta
pt_list=dgm_data[0]
pt_list[np.isinf(pt_list)] = max_death + delta 

cluster_num=s.cluster_0PD(diag,dgm_data[0])

points=p

pt_list=np.array(sorted(pt_list, key=s.custom_compare, reverse=True))

threshold=0.0
per_sup=pt_list[cluster_num-1][1]-pt_list[cluster_num-1][0]
pd_selected=[] 
for i in range(len(pt_list)):
    if pt_list[i][1]-pt_list[i][0]>=per_sup:
        pd_selected.append(pt_list[i])
threshold = round(pt_list[-1][1],6)


n = len(points)  
adjacency_matrix = csr_matrix((n, n), dtype=np.float32)  

for i in range(n):  
    for j in range(i+1, n):  
        dist = distance.euclidean(points[i], points[j])
        if dist <= threshold:
            adjacency_matrix[i, j] = dist  
            adjacency_matrix[j, i] = dist

            
plt.cla()
plt.scatter(points[:, 0], points[:, 1])   
for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
            plt.plot([points[i, 0], points[j,0]], [points[i,1], points[j, 1]], 'y-')  
plt.scatter(xr, yr,color='r')
plt.scatter(xb, yb,color='r')
plt.savefig('/results/simi1.png',dpi=300) 