import search_funcs as s
import numpy as np  
from scipy.spatial import distance
from scipy.sparse import csr_matrix  
from scipy.sparse.csgraph import depth_first_order  
from scipy.sparse.csgraph import connected_components  
import tadasets
import warnings  
warnings.filterwarnings("ignore")  
import math  
from scipy.sparse import csr_matrix  
from collections import deque
import networkx as nx
from sklearn.cluster import KMeans
import itertools
from collections import OrderedDict
import gudhi as gd
import matplotlib.pyplot as plt

def draw_all_curves(cycles,points):
    c=['orange','black','b','red']
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y,s=1)
    k=0

    for cycle_edges in cycles:
        l=len(cycle_edges)
        for p in range(l):
            line = plt.Polygon([points[cycle_edges[p%l]], points[cycle_edges[(p+1)%l]]], closed=None, fill=None, edgecolor=c[k%len(c)], linewidth=3)
            plt.gca().add_line(line)
        k=k+1
    
    plt.axis('off') 
    plt.gcf().set_size_inches(14,6) 
    plt.show()

def dfs_min_angle_path(graph, start, end, points, path=None, visited=None):
    max_iterations = len(graph.nodes)*3
    iteration_count=0
    
    if path is None:
        path = [start]
    if visited is None:
        visited = {start}

    if end in list(graph.neighbors(start)) and len(path) > 3:
        path.append(end)
        return path

    neighbors = list(graph.neighbors(start))
    if len(path)==1:
        neighbors = sorted(neighbors, key=lambda x: s.get_angle(points[end], points[start], points[x]))
    if len(path) > 1:
        neighbors = sorted(neighbors, key=lambda x: s.get_angle(points[path[-2]], points[start], points[x]))
    neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
    
    for neighbor in neighbors:
        if iteration_count > max_iterations:
            print('Reached maximum iterations')
            return None 

        iteration_count += 1 
        visited.add(neighbor)
        path.append(neighbor)
        
        result = dfs_min_angle_path(graph, neighbor, end, points, path, visited)
        if result:
            return result
        path.pop()
        visited.remove(neighbor)

    return None

def get_threshold_1PDonly(points,dgm_data,diag,Method,maxdis=0,eps=1.01):
    if Method=='PDcluster':
        loop_num=s.cluster_1PD(diag)

        pt_list=dgm_data[1]
        pt_list=np.array(sorted(pt_list, key=s.custom_compare, reverse=True))

        threshold1=-np.inf
        if len(dgm_data[1])>0:
            per_sup=pt_list[loop_num-1][1]-pt_list[loop_num-1][0] 
            pd1_selected=[]  
            for i in range(len(pt_list)):
                if pt_list[i][1]-pt_list[i][0]>=per_sup:
                    pd1_selected.append(pt_list[i])
                    threshold1=max(threshold1,pt_list[i][0])  
                    
        threshold=threshold1*eps
    return threshold, loop_num

max_dis=0.25
points=np.loadtxt('data\\4R2.txt')
plt.scatter(points[:,0], points[:,1],s=1)
plt.axis('off')
plt.gcf().set_size_inches(14,6) 
plt.show()

Method='VR'
if Method=='VR':
    rips = gd.RipsComplex(points=np.array(points), max_edge_length = max_dis)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
if Method=='alpha':
    alp = gd.AlphaComplex(points=np.array(points))
    simplex_tree = alp.create_simplex_tree()
diag = simplex_tree.persistence()
Filt = list(simplex_tree.get_filtration())

gd.plot_persistence_diagram(diag,legend=True)
plt.gcf().set_size_inches(8, 6) 
plt.show()

st=simplex_tree
indices=s.get_all_1dgms(st,points)
#print(indices)

dgm_data0=[]
dgm_data1=[]
for pd in diag:
    if pd[0]==0:
        dgm_data0.append([pd[1][0],pd[1][1]])
    if pd[0]==1:
        dgm_data1.append([pd[1][0],pd[1][1]])
dgm_data=[dgm_data0,dgm_data1]

threshold, loop_num = get_threshold_1PDonly(points,dgm_data,diag,Method='PDcluster')

print(threshold)

n = len(points)  
adjacency_matrix = csr_matrix((n, n), dtype=np.float32)   
if Method=='VR':
    for i in range(n):  
        for j in range(i+1, n):  
            dist = distance.euclidean(points[i], points[j]) 
            if dist <= threshold:  
                adjacency_matrix[i, j] = dist  
                adjacency_matrix[j, i] = dist 

if Method=='alpha':
    for f in Filt:
        if len(f[0])==2:
            i,j = f[0][0],f[0][1]
            
            if f[1] <= threshold:
                dist=distance.euclidean(points[i], points[j])
                adjacency_matrix[i,j]=dist
                adjacency_matrix[j,i]=dist
            
        
G = nx.Graph()
n=adjacency_matrix.shape[0]
for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
             G.add_edges_from([(i,j)])
             
edges=[[] for _ in range(len(indices))]
visit=[]
cycle_list=[]
k = 0
while len(cycle_list)<loop_num and k<len(indices):
    edge = sorted(indices[k])
    start_vertex, end_vertex = edge[0], edge[1]
    print(start_vertex, end_vertex)
    if adjacency_matrix[start_vertex, end_vertex]!=0:
        cycle=dfs_min_angle_path(G,start_vertex,end_vertex,points)
        #print(cycle)
        if cycle is not None:  
            #if s.not_similar(cycle,cycle_list,points):
            #s.draw_cycle(cycle, points)
            cycle_list.append(cycle)
                
    k=k+1
draw_all_curves(cycle_list,points)

'''
from PIL import Image, ImageDraw  

image_path = 'pic\\4R2\\4R2.jpg'  
image = Image.open(image_path)  
 
pixel_coords1 = cycle_list[0]
pixel_coords2 = cycle_list[1]
pixel_coords3 = cycle_list[2]
pixel_coords4 = cycle_list[3]

points1=[]
points2=[]
points3=[]
points4=[]
for i in range(len(pixel_coords1)):
    points1.append(points[pixel_coords1[i]])
for i in range(len(pixel_coords2)):
    points2.append(points[pixel_coords2[i]])    
for i in range(len(pixel_coords3)):
    points3.append(points[pixel_coords3[i]])
for i in range(len(pixel_coords4)):
    points4.append(points[pixel_coords4[i]])

max_val=3193
line1=[]
for p in points1:
    line1.append([int(p[0]*max_val),int((1-p[1])*max_val)])
line2=[]
for p in points2:
    line2.append([int(p[0]*max_val),int((1-p[1])*max_val)])
line3=[]
for p in points3:
    line3.append([int(p[0]*max_val),int((1-p[1])*max_val)])
line4=[]
for p in points4:
    line4.append([int(p[0]*max_val),int((1-p[1])*max_val)])

draw = ImageDraw.Draw(image)  

n=len(line1)
for i in range(n):  
    draw.line((line1[i%n][0],line1[i%n][1], line1[(i + 1)%n][0],line1[(i + 1)%n][1]),fill='orange', width=12)  
n=len(line2)
for i in range(n):  
    draw.line((line2[i%n][0],line2[i%n][1], line2[(i + 1)%n][0],line2[(i + 1)%n][1]),fill='black', width=12) 
n=len(line3)
for i in range(n):  
    draw.line((line3[i%n][0],line3[i%n][1], line3[(i + 1)%n][0],line3[(i + 1)%n][1]),fill='blue', width=12)  
n=len(line4)
for i in range(n):  
    draw.line((line4[i%n][0],line4[i%n][1], line4[(i + 1)%n][0],line4[(i + 1)%n][1]),fill='red', width=12) 

image.save("new_4R2.png") 

'''
