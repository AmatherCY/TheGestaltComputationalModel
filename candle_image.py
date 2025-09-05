import search_funcs as s
import numpy as np  
from scipy.spatial import distance
from scipy.sparse import csr_matrix  
from scipy.sparse.csgraph import depth_first_order  
from scipy.sparse.csgraph import connected_components  
import random
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
import ripser
import matplotlib.pyplot as plt
import cv2

max_dis=0.25

points=np.loadtxt('data//candle2.txt')
points=np.array(points)
#points=np.unique(points, axis=0) 
print(len(points))

for k in range(0,points.shape[1]):
    points[:,k]=(points[:,k]-min(points[:,k]))/(max(points[:,k])-min(points[:,k]))

plot=plt.scatter(points[:,0], points[:,1],s=5)
plt.gcf().set_size_inches(10,6) 
plt.axis('off') 
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

dgm_data0=[]
dgm_data1=[]
for pd in diag:
    if pd[0]==0:
        dgm_data0.append([pd[1][0],pd[1][1]])
    if pd[0]==1:
        dgm_data1.append([pd[1][0],pd[1][1]])
dgm_data=[dgm_data0,dgm_data1]


pt_list=dgm_data0
pt_list[0][1] = max_dis 

connect_num=s.cluster_0PD(diag,pt_list)

def custom_compare(item):  
    return item[1] - item[0] 
pt_list=np.array(sorted(pt_list, key=custom_compare, reverse=True)) 

threshold0=0.0
per_sup=pt_list[connect_num-1][1]-pt_list[connect_num-1][0] 
pd_selected=[] 
for i in range(len(pt_list)):
    if pt_list[i][1]-pt_list[i][0]>=per_sup:
        pd_selected.append(pt_list[i])

if connect_num==1:  
    threshold = pt_list[0][1]*1.01
else:
    threshold = pt_list[connect_num][1]
print(threshold)

n = len(points)  
print(n)
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

connected_components = list(nx.connected_components(G))  

p1=[]
p2=[]
p3=[]
pw1=list(connected_components[0])
pw2=list(connected_components[1])
pw3=list(connected_components[2])
for i in range(len(points)):
    if i in pw1:
        p1.append(points[i])
    elif i in pw2:
        p2.append(points[i])
    else:
        p3.append(points[i])
p1=np.array(p1)
np.savetxt('data//pc1.txt',p1,fmt='%2f')
p2=np.array(p2)
np.savetxt('data//pc2.txt',p2,fmt='%2f')
p3=np.array(p3)
np.savetxt('data//pc3.txt',p3,fmt='%2f')


plt.scatter(points[:, 0], points[:, 1],s=5)  


import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import distance

def find_and_plot_triangles(points, region_condition, color):
    selected_points = points
    tri = Delaunay(selected_points)
    valid_triangles = []

    for indices in tri.simplices:
        side_lengths = [distance.euclidean(selected_points[indices[i]], selected_points[indices[(i + 1) % 3]]) for i in range(3)]
        if all(length < 0.25 for length in side_lengths):
            valid_triangles.append(indices)

    for indices in valid_triangles:
        triangle_points = selected_points[indices]
        if np.all(region_condition(triangle_points)):
            plt.fill(triangle_points[:, 0], triangle_points[:, 1], c=color)


region10 = lambda points: (points[:, 0] < 0.09)
region11 = lambda points: (0.09 < points[:, 0]) & (points[:, 0] < 0.2)
region20 = lambda points: (0.4 < points[:, 0]) & (points[:, 0] < 0.53)
region30 = lambda points: ((0.6 < points[:, 0]) & (points[:, 0] < 0.76))
region31 = lambda points: ((0.76 < points[:, 0]) & (points[:, 0] < 0.872))
region32 = lambda points: (points[:, 0] > 0.872)


find_and_plot_triangles(points[pw1], region10, 'red')
find_and_plot_triangles(points[pw1], region11, 'red')
find_and_plot_triangles(points[pw2], region20, 'green')
find_and_plot_triangles(points[pw3], region30, 'blue')
find_and_plot_triangles(points[pw3], region31, 'blue')
find_and_plot_triangles(points[pw3], region32, 'blue')

plt.gcf().set_size_inches(10,6)  
plt.axis('off')
plt.show()  


'''
from PIL import Image, ImageDraw
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean

image_path = 'pic\\candle2\\candle2.PNG'
image = Image.open(image_path)

draw = ImageDraw.Draw(image)

points = np.loadtxt('data//candle2.txt')
width, height = image.size

def plot_delaunay_on_image(draw, points, region, color):
    selected_points = points
    selected_points[:,1]=-selected_points[:,1]
    tri = Delaunay(selected_points)

    valid_triangles = []

    for indices in tri.simplices:
        side_lengths = [euclidean(selected_points[indices[i]], selected_points[indices[(i + 1) % 3]]) for i in range(3)]
        if all(length < 100 for length in side_lengths):
            valid_triangles.append(indices)

    for indices in valid_triangles:
        triangle_points = selected_points[indices]
        triangle = [tuple(selected_points[i]) for i in indices]
        if np.all(region(triangle_points)):
            draw.polygon(triangle, fill=color, outline=color)


region10 = lambda points: (points[:, 0] < 310)
region11 = lambda points: ((310 < points[:, 0]) & (points[:, 0] < 400))
region20 = lambda points: (400 < points[:, 0]) & (points[:, 0] < 550)
region30 = lambda points: ((600 < points[:, 0]) & (points[:, 0] < 655))
region31 = lambda points: ((655 < points[:, 0]) & (points[:, 0] <= 703))
region32 = lambda points: (points[:, 0] > 703)
                         
plot_delaunay_on_image(draw, points[pw1], region10, 'red')
plot_delaunay_on_image(draw, points[pw1], region11, 'red')
plot_delaunay_on_image(draw, points[pw2], region20, 'green')
plot_delaunay_on_image(draw, points[pw3], region30, 'blue')
plot_delaunay_on_image(draw, points[pw3], region31, 'blue')
plot_delaunay_on_image(draw, points[pw3], region32, 'blue')

image.save("new_candle2.png")
'''
