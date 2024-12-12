import search_funcs as s
import matplotlib.pyplot as plt
import numpy as np  
from scipy.spatial import distance
from scipy.sparse import csr_matrix  
from scipy.sparse.csgraph import depth_first_order  
from scipy.sparse.csgraph import connected_components  
import warnings  
warnings.filterwarnings("ignore")  
import math  
from scipy.sparse import csr_matrix  
from collections import deque
import networkx as nx
from sklearn.cluster import KMeans

def draw_cycle(cycle, points):
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y,c='b')
    l = len(cycle)
    print(cycle)
    for p in range(l):
        line = plt.Line2D([points[cycle[p % l], 0], points[cycle[(p + 1) % l], 0]],
                          [points[cycle[p % l], 1], points[cycle[(p + 1) % l], 1]],
                          color='r', linewidth=2)
        plt.gca().add_line(line)
    plt.axis('equal')
    plt.gcf().set_size_inches(10, 8)
    plt.savefig('/results/goodcon.png',dpi=300)

max_dis=50
points=np.array(np.loadtxt('/data/goodcon.txt', dtype=float))

plt.scatter(points[:,0], points[:,1])
plt.xlim(-12, 26)
plt.ylim(-15, 15)
plt.gcf().set_size_inches(8, 6) 
plt.show()


import gudhi as gd
import ripser
rips = gd.RipsComplex(points=np.array(points), max_edge_length = max_dis)
simplex_tree = rips.create_simplex_tree(max_dimension=2)
diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
gd.plot_persistence_diagram(diag,legend=True)
plt.gcf().set_size_inches(8, 6) 
plt.show()
dgm_data=ripser.ripser(points)['dgms']


loop_num=s.cluster_1PD(diag)

pt_list=dgm_data[1]

pt_list=np.array(sorted(pt_list, key=s.custom_compare, reverse=True))

per_sup=pt_list[loop_num-1][1]-pt_list[loop_num-1][0]
pd1_selected=[]
for i in range(len(pt_list)):
    if pt_list[i][1]-pt_list[i][0]>=per_sup:
        pd1_selected.append(pt_list[i])
        
threshold=round(pt_list[loop_num-1][0],5)

n = len(points)  
adjacency_matrix = csr_matrix((n, n), dtype=np.float32)  


for i in range(n):  
    for j in range(i+1, n):  
        dist = distance.euclidean(points[i], points[j])
        if dist <= threshold:
            adjacency_matrix[i, j] = dist  
            adjacency_matrix[j, i] = dist

            
  
plt.scatter(points[:, 0], points[:, 1])  

for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
            plt.plot([points[i, 0], points[j,0]], [points[i,1], points[j, 1]], 'r-')  
plt.xlim(-12, 26)
plt.ylim(-15, 15)
plt.gcf().set_size_inches(8, 6) 
plt.show()  
plt.cla()

st=simplex_tree
indices=s.get_all_1dgms(st,points)
G = nx.Graph()
n=adjacency_matrix.shape[0]
for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
             G.add_edges_from([(i,j)])
print(G)

edges=[[] for _ in range(len(indices))]
visit=[]
cycle_list=[]
k = 0
while len(cycle_list)<loop_num and k<len(indices):
    edge = sorted(indices[k])
    start_vertex, end_vertex = edge[0], edge[1]
    print(start_vertex, end_vertex)
    if adjacency_matrix[start_vertex, end_vertex]!=0:
        curvature_cycle=s.dfs_min_angle_path(G,start_vertex,end_vertex,points)

        if curvature_cycle is not None:  
            if s.not_similar(curvature_cycle,cycle_list,points):
                cycle_list.append(curvature_cycle)
                draw_cycle(curvature_cycle, points)
                if len(cycle_list)>=loop_num:
                    break
    k=k+1