import math
import numpy as np
# class node(object):
#     def __init__(self,vector):
#         self.children=[]
#         self.vec=vector
#     def add_children(self,child):
#         self.children+=child
class node(object):
    def __init__(self,child,vec):
        self.children=[child]
        self.vector=vec
    def add(self,child):
        self.children+=child
def cos_similarity(n1,n2):
    array1=np.array(n1.vector)
    array2=np.array(n2.vector)
    # num = float(np.matmul(array1, array2))
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
            # return num / s
            return round(num/s,3)
    # pass
    # assert len(n1) == len(n2), "len(n1) != len(n2)"
    # zero_list = [0] * len(n1)
    # if n1 == zero_list or n2 == zero_list:
    #     return float(1) if n1 == n2 else float(0)

    # # method 1
    # res = np.array([[n1[i] * n2[i], n1[i] * n1[i], n2[i] * n2[i]] for i in range(0,60)])
    # cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    # return 0.5 * cos + 0.5 
def cluster_center_weight(current,next_node):
    current_children=len(current.children)
    next_node_children=len(next_node.children)
    sum_child=current_children+next_node_children
    weight=[round(current_children/sum_child,1),round(next_node_children/sum_child,1)]
    # new_center.append(current.vec*weight[0]+next_node.vec*weight[1])
    new_center_arr=np.array(current.vector)*weight[0]+np.array(next_node.vector)*weight[1]
    new_center=list(new_center_arr)
    return new_center
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
def cluster(data,threshold):
    initial_vec_set=load_dict(data)
    node_list=load_vec_to_node(initial_vec_set)
    # result=cluster(node_list)
    current=node_list[0]
    del node_list[0]
    result=[]
    for each_node in node_list:
        next_node=each_node
        relevancy=cos_similarity(current,next_node)
        print(relevancy)
        if relevancy>threshold:
            #for 0518 and 0512 section divided by 0.726;time divided by 0.6978
            # current.add_children(next_node.children)
            current.add(next_node.children)
            # new_children=current.children+next_node.children
            new_center=cluster_center_weight(current,next_node)
            current.vector=new_center
            # new_node=node(new_center)
            # # new_node.add_children(new_children)
            # new_node.children=new_children
            # current=new_node
        else:
            result.append(current.children)
            current=next_node
    result.append(current.children)
    # section_cluster=[]
    print(threshold)
    for index,cluster in enumerate(result):
        print('cluster {}:{}'.format(index,cluster))
        # section_cluster.append(cluster.children)
    # np.save('0512_clustered_time.npy',np.array(section_cluster))