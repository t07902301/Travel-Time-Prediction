import numpy as np
class cluster(object):
    def __init__(self,child,vec):
        self.children=[child]
        self.centroid=vec
    def add(self,child):
        self.children+=child
class node(object):
    def __init__(self,id,record,cluster_id):
        self.id=id
        self.vector=np.array(record)
        self.cluster_id=cluster_id


def cosine(n1,n2):
    array1=n1.centroid
    array2=n2.centroid
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
def pearson(n1,n2):
    return abs(np.around(np.corrcoef(n1.centroid,n2.centroid)[0][1],3))
adjacent_table={
    5:[3,55],
    3:[43],
    43:[44,385],
    44:[46,71],
    46:[138,316],
    138:[34],
    34:[32],
    32:[297,221],
    297:[152],
    152:[153],
    153:[],
    221:[42,43],
    385:[32],
    71:[57],
    316:[59],
    59:[57],
    57:[3,55],
    55:[],
    42:[153],
}
# adjacent_table={
#     5:[3],
#     3:[43],
#     43:[44],
#     44:[46],
#     46:[138],
#     138:[34],
#     34:[32],
#     32:[297],
#     297:[]
# }
nodes_dict={}
cluster_dict={}

def get_center_by_weight(current,next_node):
    current_children=len(current.children)
    next_node_children=len(next_node.children)
    sum_child=current_children+next_node_children
    weight=[round(current_children/sum_child,1),round(next_node_children/sum_child,1)]
    new_center_arr=current.centroid*weight[0]+next_node.centroid*weight[1]
    return new_center_arr
def load_dict(data):
    for id,record in data.items():
        nodes_dict[id]=node(id,record,-1)
    return data.keys()
def run(data,threshold,method):
    nodes=load_dict(data)
    cluster_num=0
    for i in nodes:
        current_node=nodes_dict[i]
        adjacent_list=adjacent_table[current_node.id]
        for j in adjacent_list:
            adjacent_node=nodes_dict[j]
            # print(current_node.cluster_id,adjacent_node.cluster_id)
            if current_node.cluster_id==-1:
                c1=cluster(current_node.id,current_node.vector)
                cluster_dict[cluster_num]=c1
                current_node.cluster_id=cluster_num
                cluster_num+=1
            else:
                c1=cluster_dict[current_node.cluster_id]
            if adjacent_node.cluster_id==-1:
                c2=cluster(adjacent_node.id,adjacent_node.vector)
                cluster_dict[cluster_num]=c2
                adjacent_node.cluster_id=cluster_num
                cluster_num+=1
            else:
                c2=cluster_dict[adjacent_node.cluster_id]
            if current_node.cluster_id==adjacent_node.cluster_id:
                continue
            if method==0:
                sim=cosine(c1,c2)
            else:
                sim=pearson(c1,c2)
            print('{} {}: {}'.format(c1.children,c2.children,sim))
            if sim>threshold:
                new_centroid=get_center_by_weight(c1,c2)
                c1.add(c2.children)
                cluster_dict[current_node.cluster_id]=c1
                cluster_num-=1
                # adjacent_node.cluster_id=current_node.cluster_id
                for child in c2.children:
                    nodes_dict[child].cluster_id=current_node.cluster_id
    result={}
    for node_id,node_info in nodes_dict.items():
        if node_info.cluster_id not in result:
            result[node_info.cluster_id]=[]
        result[node_info.cluster_id].append(node_info.id)
    result_keys=list(result.keys())
    for i in range(len(result_keys)):
        print('cluster {}: {}'.format(i,result[result_keys[i]]))
def run_random(data,threshold,method):
    nodes=load_dict(data)
    cluster_list=[]
    for node in nodes:
        cluster_list.append(cluster(nodes_dict[node].id,nodes_dict[node].vector))
    while True:
        end=len(cluster_list)
        sim_mat=[]
        for i in range(end):
            for j in range(i+1,end):
                if method==0:
                    sim=cosine(cluster_list[i],cluster_list[j])
                else:
                    sim=pearson(cluster_list[i],cluster_list[j])
                sim_mat.append((sim,[i,j]))
        if len(sim_mat)==0:
            break
        sim_max=(0,0)
        for each in sim_mat:
            if each[0]>sim_max[0]:
                sim_max=each
        print('{} and {}: {}'.format(cluster_list[sim_max[1][0]].children,cluster_list[sim_max[1][1]].children,sim_max[0]))
        if sim_max[0]<threshold:
            break
        max_pair=sim_max[1]
        new_center=get_center_by_weight(cluster_list[max_pair[0]],cluster_list[max_pair[1]])
        cluster_list[max_pair[0]].add(cluster_list[max_pair[1]].children)
        cluster_list[max_pair[0]].centroid=new_center
        del cluster_list[max_pair[1]]
        # print('{}'.format(cluster_list[max_pair[0]].children))        
    for index in range(len(cluster_list)):
        print('cluster {}: {}'.format(index,cluster_list[index].children))

# run(seg_dict,0.8,1)
# run_random(seg_dict,0.7,1)
# seg_dict={5: [0.13, 0.06, 0.06, 0.1, 0.13, 0.0, 0.0, 0.06, 0.19, 0.35, 0.61, 0.68, 0.81, 1.0], 3: [0.36, 0.36, 0.36, 0.45, 0.45, 0.0, 0.0, 0.09, 0.18, 0.45, 1.0, 1.0, 0.91, 0.82], 43: [0.0, 0.25, 0.42, 0.58, 0.67, 0.42, 0.58, 0.75, 0.92, 0.92, 1.0, 1.0, 1.0, 0.92], 44: [0.78, 0.67, 0.5, 0.39, 0.22, 0.06, 0.0, 0.11, 0.33, 0.67, 0.83, 1.0, 0.83, 0.56], 46: [0.89, 0.78, 0.78, 0.56, 0.44, 0.22, 0.0, 0.11, 0.33, 0.67, 0.89, 1.0, 0.89, 0.56], 138: [1.0, 0.85, 0.7, 0.55, 0.4, 0.15, 0.05, 0.0, 0.05, 0.15, 0.3, 0.25, 0.15, 0.0], 34: [0.0, 0.05, 0.15, 0.25, 0.35, 0.25, 0.3, 0.3, 0.45, 0.7, 0.9, 1.0, 0.8, 0.6], 32: [1.0, 0.67, 0.5, 0.33, 0.33, 0.17, 0.0, 0.17, 0.33, 0.5, 0.83, 1.0, 1.0, 0.83], 297: [0.0, 0.02, 0.05, 0.09, 0.14, 0.18, 0.23, 0.27, 0.34, 0.52, 0.7, 0.84, 0.93, 1.0]}
# run_random(seg_dict,0.7,1)
# run(seg_dict,0.7,0)

