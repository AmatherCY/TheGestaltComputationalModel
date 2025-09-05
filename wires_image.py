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
from scipy.spatial.distance import euclidean
import cv2

def draw_curve(curve,points):
    #fig, ax = plt.subplots() 
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y,s=1)
    # draw edge
    l=len(curve)
    for p in range(l-1):
        line = plt.Polygon([points[curve[p]], points[curve[(p+1)]]], closed=None, fill=None, edgecolor='r', linewidth=2)
        plt.gca().add_line(line)
    plt.axis('equal')
    plt.gcf().set_size_inches(10,8) 
    plt.show()
    
def draw_all_curves(cycles,curves,points):
    c=['g','r','y','pink','gray','b']
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y,s=1)
    k=0
    # draw edge
    if cycles:
        for cycle_edges in cycles:
            l=len(cycle_edges)
            for p in range(l):
                line = plt.Polygon([points[cycle_edges[p%l]], points[cycle_edges[(p+1)%l]]], closed=None, fill=None, edgecolor=c[k%len(c)], linewidth=4)
                plt.gca().add_line(line)
            k=k+1
    for curve_edges in curves:
        l=len(curve_edges)
        for p in range(l-1):
            line = plt.Polygon([points[curve_edges[p]], points[curve_edges[(p+1)]]], closed=None, fill=None, edgecolor=c[k%len(c)], linewidth=5)
            plt.gca().add_line(line)
        k=k+1
    
    plt.axis('off')
    plt.gcf().set_size_inches(10,8) 
    plt.show()

def find_min_angle_path(start_vertex, single_edge_vertices, points, graph):       
    path = [start_vertex]    
      
    current_vertex = start_vertex    
    count = 0  
    max_iterations = 4*len(graph.nodes) 
      
    while True:    
        if count == 0:  
            next_vertex = list(graph.neighbors(current_vertex))[1]  

        else:  
            neighbors = list(graph.neighbors(current_vertex))  
            next_vertex = s.find_min_angle_neighbor(prev_vertex, current_vertex, neighbors, points)    

        if next_vertex is not None and next_vertex not in path:    
            path.append(next_vertex)    
            prev_vertex = current_vertex    
            current_vertex = next_vertex    
        if next_vertex in single_edge_vertices or next_vertex is None: 
            print('None')
            break    
        
        if count >= max_iterations:  
            print('too much')
            return path
        count = count + 1  
    return path   

max_dis=0.05
points=np.loadtxt('data//wires2.txt')

plot=plt.scatter(points[:,0], points[:,1],s=1)
plt.gcf().set_size_inches(10,8) 
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

#gd.plot_persistence_diagram(diag,legend=True)
#plt.gcf().set_size_inches(8,6) 
#plt.show()


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

threshold = pt_list[1][1]*1.1
#threshold = pt_list[connect_num][1]
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
''' 
plt.scatter(points[:, 0], points[:, 1],s=5)  
for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
            #line = plt.Polygon([points[i, 0], points[j,0]], [points[i,1], points[j, 1]], closed=None, fill=None, edgecolor='r', linewidth=1)
            #plt.gca().add_line(line)
            plt.plot([points[i, 0], points[j,0]], [points[i,1], points[j, 1]], 'r-',linewidth=1) 
plt.gcf().set_size_inches(10,8) 
plt.show()  
'''

G = nx.Graph()
n=adjacency_matrix.shape[0]
for i in range(n):  
    for j in range(i+1, n):  
        if adjacency_matrix[i, j] != 0:  
             G.add_edges_from([(i,j)])
             
#isolate_nodes=[3863, 3300, 1398, 22]
isolate_nodes=[3774, 3255, 1398, 251]

curves=[]
while len(isolate_nodes)>0:
    curve=find_min_angle_path(isolate_nodes[0], isolate_nodes, points, G)
    curves.append(curve)
    print(curve)
    #draw_curve(curve,points)
    isolate_nodes.remove(curve[0])
    isolate_nodes.remove(curve[-1])
draw_all_curves(None,curves,points)

'''
from PIL import Image, ImageDraw  
import numpy as np

image_path = 'pic\wires2\wires2.jpg'  
image = Image.open(image_path)  

points=np.loadtxt('data\\wires2.txt')

pixel_coords1 = [3774, 3767, 3784, 3804, 3794, 3771, 3764, 3751, 3748, 3736, 3727, 3723, 3717, 3715, 3714, 3701, 3691, 3684, 3677, 3668, 3658, 3651, 3645, 3632, 3619, 3609, 3599, 3598, 3594, 3576, 3532, 3513, 3503, 3458, 3417, 3348, 3277, 3219, 3194, 3148, 3099, 3062, 3054, 3037, 3011, 2975, 2949, 2927, 2883, 2858, 2815, 2777, 2749, 2738, 2705, 2696, 2687, 2672, 2646, 2635, 2616, 2573, 2551, 2512, 2474, 2426, 2419, 2383, 2366, 2349, 2333, 2291, 2275, 2263, 2237, 2216, 2194, 2187, 2177, 2165, 2162, 2168, 2174, 2184, 2193, 2205, 2232, 2235, 2271, 2298, 2334, 2368, 2405, 2434, 2449, 2498, 2548, 2579, 2619, 2630, 2641, 2668, 2677, 2694, 2720, 2741, 2758, 2782, 2792, 2817, 2848, 2856, 2863, 2895, 2928, 2953, 2985, 3006, 3030, 3047, 3068, 3081, 3093, 3119, 3133, 3161, 3172, 3177, 3193, 3224, 3246, 3301, 3311, 3323, 3332, 3414, 3455, 3429, 3465, 3483, 3490, 3500, 3506, 3514, 3521, 3541, 3555, 3560, 3562, 3554, 3546, 3531, 3520, 3515, 3504, 3480, 3436, 3410, 3370, 3336, 3287, 3276, 3225, 3195, 3174, 3156, 3137, 3109, 3102, 3078, 3052, 3049, 3007, 2961, 2951, 2913, 2891, 2885, 2852, 2829, 2801, 2771, 2740, 2699, 2669, 2660, 2626, 2591, 2569, 2515, 2452, 2436, 2354, 2306, 2220, 2114, 2085, 2060, 2039, 2019, 1998, 1975, 1952, 1904, 1869, 1851, 1820, 1802, 1781, 1780, 1753, 1745, 1731, 1718, 1725, 1722, 1719, 1724, 1726, 1728, 1733, 1738, 1744, 1747, 1749, 1756, 1755, 1772, 1788, 1797, 1821, 1822, 1837, 1863, 1888, 1896, 1938, 1957, 1985, 1993, 2011, 2025, 2071, 2088, 2115, 2189, 2276, 2321, 2369, 2429, 2478, 2552, 2583, 2636, 2674, 2707, 2750, 2768, 2770, 2779, 2822, 2835, 2869, 2910, 2947, 2974, 2986, 2995, 3009, 3042, 3063, 3092, 3104, 3131, 3134, 3185, 3203, 3217, 3238, 3255]
#pixel_coords1 = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67] #waves
pixel_coords2 = [1398, 1362, 1347, 1312, 1302, 1299, 1284, 1249, 1207, 1119, 1111, 1054, 1032, 1012, 994, 980, 961, 949, 905, 896, 876, 832, 786, 730, 706, 628, 567, 547, 521, 497, 473, 440, 431, 418, 413, 388, 374, 361, 340, 324, 294, 278, 266, 263, 238, 217, 204, 197, 189, 173, 150, 117, 112, 110, 101, 94, 85, 81, 75, 73, 78, 74, 79, 82, 84, 93, 105, 111, 116, 146, 178, 188, 200, 213, 249, 270, 272, 301, 357, 363, 397, 417, 434, 443, 482, 505, 518, 540, 561, 569, 575, 612, 636, 711, 756, 822, 848, 889, 937, 1018, 1093, 1151, 1217, 1261, 1292, 1342, 1405, 1433, 1489, 1517, 1549, 1614, 1659, 1674, 1691, 1707, 1718, 1777, 1826, 1861, 1918, 1950, 1964, 2004, 2020, 2052, 2076, 2098, 2105, 2148, 2185, 2254, 2290, 2337, 2364, 2381, 2395, 2427, 2428, 2431, 2523, 2527, 2534, 2536, 2485, 2481, 2479, 2471, 2464, 2457, 2455, 2439, 2425, 2422, 2404, 2424, 2397, 2361, 2335, 2292, 2253, 2153, 2126, 2065, 2054, 2026, 2010, 2002, 1955, 1889, 1858, 1828, 1760, 1754, 1734, 1708, 1696, 1694, 1679, 1668, 1665, 1645, 1631, 1574, 1532, 1491, 1478, 1458, 1425, 1382, 1349, 1298, 1260, 1252, 1219, 1168, 1117, 1104, 1055, 989, 940, 919, 886, 856, 853, 843, 810, 800, 765, 752, 731, 727, 757, 733, 722, 704, 697, 694, 679, 671, 657, 649, 646, 638, 633, 634, 645, 651, 660, 677, 696, 709, 719, 745, 777, 807, 834, 854, 873, 884, 893, 916, 942, 973, 1003, 1023, 1065, 1102, 1109, 1146, 1181, 1200, 1215, 1224, 1225, 1226, 1205, 1232, 1247, 1259, 1272, 1275, 1285, 1286, 1283, 1279, 1277, 1269, 1265, 1244, 1238, 1227, 1206, 1189, 1177, 1150, 1142, 1085, 1066, 1025, 964, 939, 881, 842, 809, 740, 716, 689, 666, 577, 553, 549, 537, 534, 522, 517, 489, 477, 428, 389, 364, 306, 298, 251]

points1=[]
points2=[]
for i in range(len(pixel_coords1)):
    points1.append(points[pixel_coords1[i]])
for i in range(len(pixel_coords2)):
    points2.append(points[pixel_coords2[i]])    

inp=3203
line1=[]
for p in points1:
    line1.append([int(p[0]*inp),int((1-p[1])*inp)])
line2=[]
for p in points2:
    line2.append([int(p[0]*inp),int((1-p[1])*inp)])
line3=[]


draw = ImageDraw.Draw(image)  
  
n=len(line1)
for i in range(n-1):  
    draw.line((line1[i][0],line1[i][1], line1[(i + 1)][0],line1[(i + 1)][1]),fill='green', width=15)  
n=len(line2)
for i in range(n-1):  
    draw.line((line2[i][0],line2[i][1], line2[(i + 1)][0],line2[(i + 1)][1]),fill='red', width=15) 
 
image.save("new_wires2.png") 

'''
