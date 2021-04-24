import time
import numpy as np
class node(object):
    def __init__(self,child,vec):
        self.children=[child]
        self.vector=np.array(vec)
    def add(self,child):
        self.children+=child
def pearson(n1,n2):
    return np.around(np.corrcoef(n1.vector,n2.vector)[0][1],3)
def cos_similarity(n1,n2):
    array1=n1.vector
    array2=n2.vector
    try:
        num=np.dot(array1,array2)
    except ValueError as error:
        print(n1.children)
        print(n2.children)
        return 0.0
    except TypeError as te:
        print(n1.children,n2.children)
        return 0.0
    else:
        s = np.linalg.norm(array1) * np.linalg.norm(array2)
        if s==0:
            return 0.0
        else:
            return round(num/s,3)
def get_center_by_weight(current,next_node):
    current_children=len(current.children)
    next_node_children=len(next_node.children)
    sum_child=current_children+next_node_children
    weight=[round(current_children/sum_child,1),round(next_node_children/sum_child,1)]
    new_center_arr=current.vector*weight[0]+next_node.vector*weight[1]
    return new_center_arr
def load_dict(data):
    vector_set=[]
    for section,record in data.items():
        vector_set.append([section,record])
    return vector_set   
def load_vec_to_node(vec_set):
    node_list=[]
    for vec in vec_set:
        node_list.append(node(vec[0],vec[1]))
    return node_list

def cluster_with_cosine(data,threshold):
    initial_vec_set=load_dict(data)
    node_list=load_vec_to_node(initial_vec_set)
    current=node_list[0]
    del node_list[0]
    result=[]
    for each_node in node_list:
        next_node=each_node
        relevancy=cos_similarity(current,next_node)
        print(current.children,next_node.children,relevancy)
        if relevancy>threshold:
            current.add(next_node.children)
            new_center=get_center_by_weight(current,next_node)
            current.vector=new_center
        else:
            result.append(current.children)
            current=next_node
    result.append(current.children)
    print(threshold)
    for index,cluster in enumerate(result):
        print('cluster {}:{}'.format(index,cluster))

def cluster_pearson(data,threshold):
    initial_vec_set=load_dict(data)
    node_list=load_vec_to_node(initial_vec_set)
    current=node_list[0]
    del node_list[0]
    result=[]
    for each_node in node_list:
        next_node=each_node
        relevancy=pearson(current,next_node)
        print(current.children,next_node.children,relevancy)
        if abs(relevancy)>threshold:
            current.add(next_node.children)
            new_center=get_center_by_weight(current,next_node)
            current.vector=new_center
        else:
            result.append(current.children)
            current=next_node
    result.append(current.children)
    print(threshold)
    for index,cluster in enumerate(result):
        print('cluster {}:{}'.format(index,cluster))

def cluster_with_cosine_init(data,threshold):
    initial_vec_set=load_dict(data)
    node_list=load_vec_to_node(initial_vec_set)
    while True:
        end=len(node_list)
        sim_mat=[]
        for i in range(end):
            for j in range(i+1,end):
                sim_mat.append((cos_similarity(node_list[i],node_list[j]),[i,j]))
        if len(sim_mat)==0:
            break
        sim_max=(0,0)
        for each in sim_mat:
            if each[0]>sim_max[0]:
                sim_max=each
        print('{} and {}: {}'.format(node_list[sim_max[1][0]].children,node_list[sim_max[1][1]].children,sim_max[0]))
        if sim_max[0]<threshold:
            break
        max_pair=sim_max[1]
        new_center=get_center_by_weight(node_list[max_pair[0]],node_list[max_pair[1]])
        node_list[max_pair[0]].add(node_list[max_pair[1]].children)
        node_list[max_pair[0]].vector=new_center
        del node_list[max_pair[1]]
        print('{}'.format(node_list[max_pair[0]].children))

    for index,cluster in enumerate(node_list):
        print('cluster {}:{}'.format(index,cluster.children))

def cluster_with_pearson_init(data,threshold):
    initial_vec_set=load_dict(data)
    node_list=load_vec_to_node(initial_vec_set)
    while True:
        end=len(node_list)
        sim_mat=[]
        for i in range(end):
            for j in range(i+1,end):
                sim_mat.append((abs(pearson(node_list[i],node_list[j])),[i,j]))
        sim_max=(0,0)
        for each in sim_mat:
            if each[0]>sim_max[0]:
                sim_max=each
        print('{} and {}: {}'.format(node_list[sim_max[1][0]].children,node_list[sim_max[1][1]].children,sim_max[0]))
        if sim_max[0]<threshold:
            break
        max_pair=sim_max[1]
        new_center=get_center_by_weight(node_list[max_pair[0]],node_list[max_pair[1]])
        new_cluster=node(node_list[max_pair[0]].add(node_list[max_pair[1]].children),new_center)
        del node_list[max_pair[1]]
        print('{}'.format(node_list[max_pair[0]].children))

    for index,cluster in enumerate(node_list):
        print('cluster {}:{}'.format(index,cluster.children))
seg_dict={5: [0.19, 0.17, 0.17, 0.19, 0.19, 0.14, 0.15, 0.18, 0.21, 0.28, 0.35, 0.37, 0.41, 0.46], 3: [0.27, 0.27, 0.27, 0.29, 0.28, 0.23, 0.24, 0.25, 0.26, 0.31, 0.37, 0.36, 0.35, 0.34], 43: [0.3, 0.33, 0.35, 0.37, 0.37, 0.34, 0.36, 0.39, 0.4, 0.41, 0.42, 0.42, 0.42, 0.4], 44: [0.69, 0.67, 0.64, 0.62, 0.59, 0.56, 0.56, 0.58, 0.63, 0.68, 0.72, 0.74, 0.7, 0.65], 46: [0.52, 0.52, 0.51, 0.5, 0.48, 0.45, 0.44, 0.45, 0.47, 0.51, 0.54, 0.54, 0.52, 0.5], 138: [0.31, 0.28, 0.24, 0.22, 0.18, 0.13, 0.11, 0.1, 0.11, 0.13, 0.16, 0.14, 0.12, 0.1], 34: [0.28, 0.28, 0.29, 0.3, 0.31, 0.3, 0.3, 0.32, 0.35, 0.39, 0.43, 0.44, 0.4, 0.36], 32: [0.3, 0.28, 0.27, 0.26, 0.26, 0.24, 0.24, 0.24, 0.26, 0.28, 0.3, 0.31, 0.3, 0.29], 297: [0.2, 0.21, 0.22, 0.23, 0.26, 0.28, 0.29, 0.31, 0.35, 0.44, 0.52, 0.58, 0.62, 0.64]}
start=time.time()
cluster_with_pearson_init(seg_dict,0.7)
end=time.time()
print(end-start)