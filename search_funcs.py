import matplotlib.pyplot as plt
import numpy as np  
from scipy.spatial import distance
from scipy.sparse import csr_matrix   
import warnings   
import math  
from scipy.sparse import csr_matrix  
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import deque
import networkx as nx


def custom_compare(item):  
    return item[1] - item[0]
 
def get_angle(p0, p1, p2):  
        dx1 = p1[0] - p0[0]  
        dy1 = p1[1] - p0[1]  
        dx2 = p2[0] - p1[0]  
        dy2 = p2[1] - p1[1]  
        angle1 = math.atan2(dy1, dx1)  
        angle2 = math.atan2(dy2, dx2)  
        angle = (angle2 - angle1) % (2 * math.pi)  
        if angle > math.pi:  
            angle -= 2 * math.pi  
        return abs(angle)  
def calculate_angle(x0, x1, n, points):  
    p0 = points[x0]  
    p1 = points[x1]  
    p2 = points[n]  
    return get_angle(p0, p1, p2)  

def cluster_0PD(diag,dgm_data):
    data = list()
    for pd in diag:
        if pd[0] == 0:
            data.append(pd[1])
    if len(data)==1:
        y=[0]
    else:
        data = list()
        maxn=0
        for i in range(len(dgm_data)):
            num = (dgm_data[i][0] - dgm_data[i][1]) / 2.0
            maxn=max(maxn,-num)
            data.append([num, -num])
        lendata=len(data)
        for j in range(len(data)):
            data.append([0,0])
        data.sort(key=custom_compare,reverse=True)
        initial_centroids=np.array([[-maxn,maxn],[0,0]])
        k_means = KMeans(n_clusters=2,init=initial_centroids,n_init=1)
        k_means.fit(data)
        y = list(k_means.predict(data))
        
    cluster_num=0

    if data[0][1]>data[-1][1]: 
        for e in y:  
            if e==y[0]:
                cluster_num=cluster_num+1
    else:
        for e in y:  
            if e==y[-1]:
                cluster_num=cluster_num+1
  
    if cluster_num== 0.5*len(data):
        cluster_num=1
    print('cluster0_num = '+str(cluster_num))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    silhouette = silhouette_score(data, kmeans.labels_)
    print(f"Silhouette Score: {silhouette:.3f}")
    return cluster_num

def cluster_1PD(diag,thres=0): 
    data = list()
    for pd in diag:
        if pd[0] == 1:
            data.append(pd[1])
    if len(data)==0:
        return 0
    
    if len(data)==1:
        pd=data[0]
        if pd[1]-pd[0]>thres:
            print('cluster1_num = 1')
            return 1
        else:
            print('cluster1_num = 0')
            return 0
    
    data = list()
    
    maxn=0
    for pd in diag:
        if pd[0] == 1:
            num = (pd[1][0] - pd[1][1]) / 2.0
            maxn=max(maxn,-num)
            data.append([num,-num])
    lendata=len(data)
    
    for j in range(len(data)):
        data.append([0,0])
    data.sort(key=custom_compare,reverse=True)
    initial_centroids=np.array([[-maxn,maxn],[0,0]])
    k_means = KMeans(n_clusters=2,init=initial_centroids,n_init=1)
    k_means.fit(data)
    y = list(k_means.predict(data))

    cluster1_num=0
    if data[0][1]>data[-1][1]:
        for e in y:  
            if e==y[0]:
                cluster1_num=cluster1_num+1
    else:
        for e in y:  
            if e==y[-1]:
                cluster1_num=cluster1_num+1

    print('cluster1_num ='+str(cluster1_num))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    silhouette = silhouette_score(data, kmeans.labels_)
    print(f"Silhouette Score: {silhouette:.3f}")
    return cluster1_num

def dfs_min_angle_path(graph, start, end, points, path=None, visited=None):
    if path is None:
        path = [start]
    if visited is None:
        visited = {start}

    if start == end and len(path) >= 3:
        return path

    neighbors = list(graph.neighbors(start))
    if len(path)==1:
        neighbors = sorted(neighbors, key=lambda x: get_angle(points[end], points[start], points[x]))
    if len(path) > 1:
        neighbors = sorted(neighbors, key=lambda x: get_angle(points[path[-2]], points[start], points[x]))
    neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]

    for neighbor in neighbors:
        visited.add(neighbor)
        path.append(neighbor)
        result = dfs_min_angle_path(graph, neighbor, end, points, path, visited)
        if result:
            return result
        path.pop()
        visited.remove(neighbor)
    
    return None


def find_min_angle_neighbor(prev_node, node, neighbors, points, exclude=None):
    # if exclude is not None, we exclude it from the neighbors
    if exclude is not None and exclude in neighbors:
        neighbors.remove(exclude)
    angles = [get_angle(points[prev_node], points[node], points[n]) for n in neighbors]
    # sort neighbors based on angles
    sorted_neighbors = [x for _, x in sorted(zip(angles, neighbors))]
    return sorted_neighbors[0] if len(sorted_neighbors) > 0 else None

def find_min_angle_path(start_vertex, single_edge_vertices, points, graph):    
    path = [start_vertex]  

    current_vertex = start_vertex  
    count=0
    while True:  
        if count==0:
            count=count+1
            next_vertex=list(graph.neighbors(current_vertex))[0]
        else:
            neighbors= list(graph.neighbors(current_vertex))
            next_vertex = find_min_angle_neighbor(prev_vertex, current_vertex, neighbors, points)  
            
        if next_vertex is not None and next_vertex != start_vertex:  
            path.append(next_vertex)  
            prev_vertex = current_vertex  
            current_vertex = next_vertex  
        if next_vertex in single_edge_vertices:
            path.append(next_vertex)  
            break  
  
    return path  

def draw_cycle(cycle,points):
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y)
    l=len(cycle)
    print(cycle)
    for p in range(l):
        line = plt.Polygon([points[cycle[p%l]], points[cycle[(p+1)%l]]], closed=None, fill=None, edgecolor='r', linewidth=2)
        plt.gca().add_line(line)
    plt.axis('equal')
    plt.gcf().set_size_inches(10,8) 
    plt.show()

    
def get_all_1dgms(st,points):
    st.compute_persistence()
    persistence = st.persistence_pairs()

    indices=[]
    for ps in persistence:
        if len(ps[0])==2 and len(ps[1])==3:
            birth=distance.euclidean(points[ps[0][0]],points[ps[0][1]])
            death=max(distance.euclidean(points[ps[1][0]],points[ps[1][1]]),distance.euclidean(points[ps[1][0]],points[ps[1][2]]))
            death=max(death,distance.euclidean(points[ps[1][1]],points[ps[1][2]]))
            pers=death-birth
            
            indices.append([ps[0][0],ps[0][1],pers])

    def cmp(x):
        return x[2]
    indices.sort(key=cmp,reverse=True)
    indices=[sublist[:2] for sublist in indices]
    return indices

def euclidean_distance(p1, p2):  
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  
  
def hausdorff_distance(points, line1, line2):
    max_distance = 0  
    for index1 in line1:  
        point1 = points[index1]  
        distances = [euclidean_distance(point1, points[index2]) for index2 in line2]  
        min_distance = min(distances)  
        if min_distance > max_distance:  
            max_distance = min_distance  
    for index2 in line2:  
        point2 = points[index2]  
        distances = [euclidean_distance(point2, points[index1]) for index1 in line1]  
        min_distance = min(distances)  
        if min_distance > max_distance:  
            max_distance = min_distance  
    return max_distance  

def jaccard_similarity(points, line1, line2):
    set1 = set(line1)  
    set2 = set(line2)  
    intersection = len(set1.intersection(set2))  
    union = len(set1.union(set2))  
    jaccard_similarity = intersection / union  
    return jaccard_similarity  

def not_similar(cycle,cycle_list,points):
    for c in cycle_list:
        if hausdorff_distance(points,cycle,c)<1 or jaccard_similarity(points,cycle,c)>0.5:  
            return False
    return True
    

def get_threshold_1PDonly(points,dgm_data,diag,Method,maxdis=0,eps=1.01):
    if Method=='PDcluster':
        # 聚类得到的重要的环的个数
        loop_num=cluster_1PD(diag)

        pt_list=dgm_data[1]
        pt_list=np.array(sorted(pt_list, key=custom_compare, reverse=True)) #按persistence从大到小排序

        threshold1=-np.inf
        if len(dgm_data[1])>0:
            per_sup=pt_list[loop_num-1][1]-pt_list[loop_num-1][0] #阈值的选取正好能够分出前loop_num个持续时间最大的PD
            pd1_selected=[]  #筛选出持续时间大于 per_sup 的1PD
            for i in range(len(pt_list)):
                if pt_list[i][1]-pt_list[i][0]>=per_sup:
                    pd1_selected.append(pt_list[i])
                    threshold1=max(threshold1,pt_list[i][0])  #阈值取1维PD点里面出生时间最大的值
                    
        threshold=threshold1*eps
    return threshold, loop_num

def find_single_edge_vertices(adj_matrix):  
    # 初始化一个空列表来存储只与一条边相连的顶点的索引  
    single_edge_vertices = []  
  
    # 遍历每个顶点  
    for i in range(adj_matrix.shape[0]):  
        
        num_connected_vertices = 0
        # 找出与当前顶点相连的顶点数量  
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:
                num_connected_vertices = num_connected_vertices + 1
  
        # 如果只有一个相连的顶点，将当前顶点的索引添加到结果列表中  
        if num_connected_vertices == 1:  
            single_edge_vertices.append(i)  
  
    return single_edge_vertices 

def draw_curve(curve,points):
    #绘制非闭曲线
    #fig, ax = plt.subplots() 
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y)
    # draw edge
    l=len(curve)
    for p in range(l-1):
        line = plt.Polygon([points[curve[p]], points[curve[(p+1)]]], closed=None, fill=None, edgecolor='r', linewidth=2)
        plt.gca().add_line(line)
    plt.axis('equal')
    plt.gcf().set_size_inches(10,8) 
    plt.show()