import numpy as np
class cluster(object):
    def __init__(self,child,vec):
        self.children=[child]
        self.centroid=vec
        # self.id=id
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
def pearson(v1,v2):
    return np.around(np.corrcoef(v1,v2)[0][1],3)

nodes_dict={}
cluster_dict={}
sim_mat={}
# [139,316,46,47,138,33,34,45,246,385,42,37,41,43,44,152]
adjacent_table={
    139:[],
    316:[47,138],
    46:[138,139],
    47:[45],
    138:[34],
    33:[],
    34:[],
    45:[385,246],
    246:[42],
    41:[43],
    43:[44,385],
    44:[46],
    152:[41],
    37:[41],
    42:[],
    385:[33]
}
def get_center_by_weight(current,next_node):
    current_children=len(current.children)
    next_node_children=len(next_node.children)
    sum_child=current_children+next_node_children
    weight=[round(current_children/sum_child,1),round(next_node_children/sum_child,1)]
    new_center_arr=current.centroid*weight[0]+next_node.centroid*weight[1]
    return new_center_arr
def load_dict(data):
    # node_id=0
    for id,record in data.items():
        nodes_dict[id]=node(id,record,-1)
    data_keys=list(data.keys())
    data_keys.sort()
    return data_keys
def generate_sim_mat(keys):
    keys_num=len(keys)
    for i in range(keys_num):
        sim_mat_key={}
        for j in range(i,keys_num):
            sim_mat_key[keys[j]]=pearson(nodes_dict[keys[i]].vector,nodes_dict[keys[j]].vector)
            # sim_mat_key[keys[j]]=10
        sim_mat[keys[i]]=sim_mat_key
def dump_sim_mat(keys):
    keys_num=len(keys)
    affinity_mat=[]
    for i in range(keys_num):
        row=[]
        for j in range(i):
            sim=pearson(nodes_dict[keys[i]].vector,nodes_dict[keys[j]].vector)
            if sim<0:
                sim=0
            row.append(sim)
        affinity_mat.append(row)
    for i in range(keys_num):
        row=affinity_mat[i]
        row.append(1)
        for j in range(i+1,keys_num):
            row.append(affinity_mat[j][i])
        affinity_mat[i]=row
    return affinity_mat
def single_linkage(c1,c2):
    min_sim=1
    for n1 in c1.children:
        for n2 in c2.children:
            if n2 in sim_mat[n1]:
                tmp_sim=sim_mat[n1][n2]
            else:
                tmp_sim=sim_mat[n2][n1]
            if min_sim>tmp_sim:
                min_sim=tmp_sim 
    return min_sim
def average_linkage(c1,c2):
    sum_sim=0
    for n1 in c1.children:
        for n2 in c2.children:
            if n2 in sim_mat[n1]:
                sum_sim+=sim_mat[n1][n2]
            else:
                sum_sim+=sim_mat[n2][n1]
    return round(sum_sim/(len(c1.children)*len(c2.children)),2)

def hierarchical(data,linkage,method,threshold):
    '''
    linkage: 0-centroid,1-average,2-single\n
    method: 0-similarity-based,1-cluster number-based\n
    threshold
    '''
    # linkage_mat=[]
    nodes=load_dict(data)
    cluster_list=[]
    if linkage>0:
        generate_sim_mat(nodes)
    for index,node in enumerate(nodes):
        cluster_list.append(cluster(nodes_dict[node].id,nodes_dict[node].vector))
    # cluster_num=len(cluster_list)-1
    while True:
        end=len(cluster_list)
        tmp_sim_mat=[]
        for i in range(end):
            for j in range(i+1,end):
                if linkage==1:
                    sim=average_linkage(cluster_list[i],cluster_list[j])
                elif linkage==2:
                    sim=single_linkage(cluster_list[i],cluster_list[j])
                else:
                    sim=pearson(cluster_list[i].centroid,cluster_list[j].centroid)                
                tmp_sim_mat.append((sim,[i,j]))
        if len(tmp_sim_mat)==0:
            break
        sim_max=(0,0)
        for each in tmp_sim_mat:
            if each[0]>sim_max[0]:
                sim_max=each
        print('{} and {}: {}'.format(cluster_list[sim_max[1][0]].children,cluster_list[sim_max[1][1]].children,sim_max[0]))
        if method==0:# based on similarity
            if sim_max[0]<threshold:
                break
        else:
            if len(cluster_list)==threshold:
                break
        max_pair=sim_max[1]

        # print('{},{},{},{}'.format(cluster_list[max_pair[0]].id,cluster_list[max_pair[1]].id,sim_max[0],len(cluster_list[max_pair[0]].children)+len(cluster_list[max_pair[1]].children)))
        # cluster_num+=1
        # linkage_mat.append([cluster_list[max_pair[0]].id,cluster_list[max_pair[1]].id,sim_max[0],len(cluster_list[max_pair[0]].children)+len(cluster_list[max_pair[1]].children),cluster_num])
        # cluster_list[max_pair[0]].id=cluster_num

        cluster_list[max_pair[0]].add(cluster_list[max_pair[1]].children)
        if linkage==0:
            new_center=get_center_by_weight(cluster_list[max_pair[0]],cluster_list[max_pair[1]])
            cluster_list[max_pair[0]].centroid=new_center
        del cluster_list[max_pair[1]]
        
        # print('{}'.format(cluster_list[max_pair[0]].children))   
    for index in range(len(cluster_list)):
        print('cluster {}: {}'.format(index,cluster_list[index].children))
    # print(linkage_mat)
    print([cluster_list[index].children for index in range(len(cluster_list))])
def sfhc(data,linkage,threshold):
    '''
    linkage: 0-centroid,1-average,2-single\n
    threshold
    '''    
    nodes=load_dict(data)
    cluster_num=0
    if linkage>0:
        generate_sim_mat(nodes)
    for i in nodes:
        current_node=nodes_dict[i]
        adjacent_list=adjacent_table[current_node.id]
        for j in adjacent_list:
            adjacent_node=nodes_dict[j]
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
            if linkage==1:
                sim=average_linkage(c1,c2)
            elif linkage==2:
                sim=single_linkage(c1,c2)
            else:#linkage=0
                sim=pearson(c1.centroid,c2.centroid)
            print('{} {}: {}'.format(c1.children,c2.children,sim))
            if sim>threshold:
                c1.add(c2.children)
                if linkage==0:
                    new_centroid=get_center_by_weight(c1,c2)
                    c1.centroid=new_centroid
                cluster_dict[current_node.cluster_id]=c1

                for child in c2.children:
                    nodes_dict[child].cluster_id=current_node.cluster_id
    result={}
    for node_id,node_info in nodes_dict.items():
        if node_info.cluster_id not in result:
            result[node_info.cluster_id]=[]
        result[node_info.cluster_id].append(node_id)
    result_keys=list(result.keys())
    for i in range(len(result_keys)):
        print('cluster {}: {}'.format(i,result[result_keys[i]]))
    print([result[result_keys[i]] for i in range(len(result_keys))])
def dump(data):
    nodes=load_dict(data)
    affinity=dump_sim_mat(nodes)
    # print(affinity)
    return affinity