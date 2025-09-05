import matplotlib.pyplot as plt
import numpy as np  
from scipy.spatial import distance
from scipy.sparse import csr_matrix  
from scipy.sparse.csgraph import depth_first_order  
from scipy.sparse.csgraph import connected_components  
import search_funcs as s
import warnings  
warnings.filterwarnings("ignore")  
import math  
from scipy.sparse import csr_matrix  
import gudhi as gd
import networkx as nx

#step_size = 20

max_dis=10
p = np.loadtxt('data/waves111.txt')

for k in range(0,p.shape[1]):
    if k>1:
        p[:,k]=max_dis * (p[:,k] - np.min(p[:,k])) / (np.max(p[:,k]) - np.min(p[:,k]))
    else:
        p[:,k]=(p[:,k]-min(p[:,k]))/(max(p[:,k])-min(p[:,k]))
for k in range(0,p.shape[0]):
    p[k,1]=1-p[k,1]

     
plot=plt.scatter(p[:,0], p[:,1], c=p[:,2])
plt.colorbar(plot)
plt.gcf().set_size_inches(8,6) 
plt.axis('equal')
plt.show()

import gudhi as gd
#alp = gd.AlphaComplex(points=np.array(p))
#simplex_tree = alp.create_simplex_tree()
rips = gd.RipsComplex(points=np.array(p), max_edge_length = max_dis)
simplex_tree = rips.create_simplex_tree(max_dimension=2)
diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

gd.plot_persistence_diagram(diag,legend=True)
plt.gcf().set_size_inches(8, 6) 
plt.show() 

import ripser
dgm_data=ripser.ripser(p)['dgms']

# Treating infinity points by borrowing from GUDHI's method of plotting PDs
points=p
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

pt_list=np.array(sorted(pt_list, key=s.custom_compare, reverse=True))

threshold=0.0
per_sup=pt_list[cluster_num-1][1]-pt_list[cluster_num-1][0]
pd_selected=[]
for i in range(len(pt_list)):
    if pt_list[i][1]-pt_list[i][0]>=per_sup:
        pd_selected.append(pt_list[i])
if cluster_num == 1:
    threshold = pt_list[1][1]
else:
    threshold = pt_list[cluster_num][1]
print(threshold)

n = len(points)  
adjacency_matrix = csr_matrix((n, n), dtype=np.float32)  
for i in range(n):  
    for j in range(i+1, n):  
        dist = distance.euclidean(points[i], points[j])
        if dist <= threshold:
            adjacency_matrix[i, j] = dist
            adjacency_matrix[j, i] = dist
G = nx.Graph()
n=adjacency_matrix.shape[0]
for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
             G.add_edges_from([(i,j)])

   
connected_components = list(nx.connected_components(G))      
            
pw1=[]
pw2=[]
p1=[]
p2=[]
pw1=list(connected_components[0])
pw2=list(connected_components[1])

for i in range(len(points)):
    if i in pw1:
        p1.append(points[i])
    else:
        p2.append(points[i])

p1=np.array(p1)
np.savetxt('data\\pw1.txt',p1,fmt='%2f')
p2=np.array(p2)
np.savetxt('data\\pw2.txt',p2,fmt='%2f')
  
from mpl_toolkits.mplot3d import Axes3D  
  
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  
  
ax.scatter(points[:, 0], points[:, 1], points[:, 2])  
  
for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
            ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], [points[i, 2], points[j, 2]], 'y-')  
plt.show()  
