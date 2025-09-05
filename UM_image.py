import matplotlib.pyplot as plt
import numpy as np  
from scipy.spatial import distance
from scipy.sparse import csr_matrix  
from scipy.sparse.csgraph import depth_first_order  
from scipy.sparse.csgraph import connected_components  
import search_funcs as s
import tadasets
import warnings  
warnings.filterwarnings("ignore")  
import math  
from scipy.sparse import csr_matrix  
import gudhi as gd
import networkx as nx
import time

def draw_all_curves(cycles,points):
    c=['r','g','b','pink','orange']
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y,s=1)
    k=0

    for cycle_edges in cycles:
        l=len(cycle_edges)
        for p in range(l):
            line = plt.Polygon([points[cycle_edges[p%l]], points[cycle_edges[(p+1)%l]]], closed=None, fill=None, edgecolor=c[k%len(c)], linewidth=2)
            plt.gca().add_line(line)
        k=k+1

    
    plt.axis('off')  # 关闭坐标轴
    plt.gcf().set_size_inches(8,6) 
    plt.show()

max_dis=10
p = np.loadtxt('data\\UM1.txt')

for k in range(0,p.shape[1]): #数据归一化
    if k>1:
        p[:,k]=max_dis * (p[:,k] - np.min(p[:,k])) / (np.max(p[:,k]) - np.min(p[:,k]))
    else:
        p[:,k]=(p[:,k]-min(p[:,k]))/(max(p[:,k])-min(p[:,k]))
for k in range(0,p.shape[0]):
    p[k,1]=1-p[k,1]
for k in range(2,p.shape[1]):       
    for i in range(p.shape[0]):
        p[i,k]= max_dis if p[i,k]>max_dis*0.5 else 0
'''     
plot=plt.scatter(p[:,0], p[:,1], c=p[:,2])
plt.colorbar(plot)
plt.gcf().set_size_inches(8,6) 
plt.axis('equal')
plt.show()
'''
import gudhi as gd
#alp = gd.AlphaComplex(points=np.array(p))
#simplex_tree = alp.create_simplex_tree()
rips = gd.RipsComplex(points=np.array(p), max_edge_length = max_dis+1)
simplex_tree = rips.create_simplex_tree(max_dimension=2)
diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

'''
gd.plot_persistence_diagram(diag,legend=True)
plt.gcf().set_size_inches(8, 6) 
plt.show() 
'''

start = time.time()
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
print(dgm_data[0][:cluster_num])

pt_list=np.array(sorted(pt_list, key=s.custom_compare, reverse=True))
threshold1, loop_num = s.get_threshold_1PDonly(points,dgm_data,diag,Method='PDcluster')

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
threshold=max(threshold,threshold1)*1.01
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
np.savetxt('data\\pu1.txt',p1,fmt='%2f')
p2=np.array(p2)
np.savetxt('data\\pu2.txt',p2,fmt='%2f')

#3d 图像       
from mpl_toolkits.mplot3d import Axes3D  
'''
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  
  
ax.scatter(points[:, 0], points[:, 1], points[:, 2])  
  
for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
            ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], [points[i, 2], points[j, 2]], 'y-')  
plt.show()  
'''


points=np.loadtxt('data\\pu2.txt')[:,:2]
'''
plot=plt.scatter(points[:,0], points[:,1])
plt.gcf().set_size_inches(8,6) 
plt.axis('equal')
plt.show()
'''

Method='VR'

if Method=='VR':
    rips = gd.RipsComplex(points=np.array(points), max_edge_length = max_dis)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
if Method=='alpha':
    alp = gd.AlphaComplex(points=np.array(points))
    simplex_tree = alp.create_simplex_tree()
diag = simplex_tree.persistence()
Filt = list(simplex_tree.get_filtration())
'''
gd.plot_persistence_diagram(diag,legend=True)
plt.gcf().set_size_inches(8, 6) 
plt.show()
'''
#获取环的指标
st=simplex_tree
indices=s.get_all_1dgms(st,points)
#print(indices)

#普通持续图
dgm_data0=[]
dgm_data1=[]
for pd in diag:
    if pd[0]==0:
        dgm_data0.append([pd[1][0],pd[1][1]])
    if pd[0]==1:
        dgm_data1.append([pd[1][0],pd[1][1]])
dgm_data=[dgm_data0,dgm_data1]
print(dgm_data[1])
threshold, loop_num = s.get_threshold_1PDonly(points,dgm_data,diag,Method='PDcluster')
print(threshold)

n = len(points)  
adjacency_matrix = csr_matrix((n, n), dtype=np.float32)   
if Method=='VR':
    for i in range(n):  
        for j in range(i+1, n):  
            dist = distance.euclidean(points[i], points[j])  # 假设使用欧氏距离  
            if dist <= threshold:  # 根据权重阈值来判断是否连接  
                adjacency_matrix[i, j] = dist  
                adjacency_matrix[j, i] = dist  # 无向图，对称矩阵

if Method=='alpha':
    for f in Filt:
        if len(f[0])==2:
            i,j = f[0][0],f[0][1]
            
            if f[1] <= threshold:  # 根据权重阈值来判断是否连接 
                dist=distance.euclidean(points[i], points[j])
                adjacency_matrix[i,j]=dist
                adjacency_matrix[j,i]=dist
            
        
G = nx.Graph()
n=adjacency_matrix.shape[0]
for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
             G.add_edges_from([(i,j)])

#最小转角算法
edges=[[] for _ in range(len(indices))]
visit=[]
cycle_list=[]
k = 0
while len(cycle_list)<loop_num and k<len(indices): #找满loop_num个环
    edge = sorted(indices[k])
    start_vertex, end_vertex = edge[0], edge[1]
    print(start_vertex, end_vertex)
    if adjacency_matrix[start_vertex, end_vertex]!=0:
        cycle=s.dfs_min_angle_path(G,start_vertex,end_vertex,points)
        #print(cycle)
        if cycle is not None:  
            #if s.not_similar(cycle,cycle_list,points,haus_thres=0.05):  #用Hausdorff距离判断两个环是否相似
            cycle_list.append(cycle)
                          
    k=k+1
#draw_all_curves(cycle_list,points)
end = time.time()
print(f"Time: {end - start}s")