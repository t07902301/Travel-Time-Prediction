import random
import numpy
import tensorflow as tf

def evaluate(model,targets,test_model_in,test_model_out):
    max_arr=numpy.array([ seg_max_min[seg][0] for seg in targets])
    min_arr=numpy.array([seg_max_min[seg][1] for seg in targets])
    bias_arr=max_arr-min_arr#(路段0，路段1,...)
    # bias_arr_expanded=numpy.array([bias_arr]*test_model_in[0].shape[0])
    min_arr_expanded=numpy.array([min_arr]*test_model_in[0].shape[0])
    all_pred=model.predict(test_model_in)
    pred=all_pred*bias_arr+min_arr_expanded 
    gt=(test_model_out[0]*bias_arr+min_arr_expanded)
    mae_tmp=numpy.absolute(pred-gt)
    mape_tmp=mae_tmp/gt
    return mape_tmp,gt,pred
def evaluate_stdn(model,targets,test_model_in,test_model_out):
    max_arr=numpy.array([ seg_max_min_stdn[seg][0] for seg in targets])# STDN 模型对应的最大最小值范围和其他模型不同
    min_arr=numpy.array([seg_max_min_stdn[seg][1] for seg in targets])
    bias_arr=max_arr-min_arr#(路段0，路段1,...)
    # bias_arr_expanded=numpy.array([bias_arr]*test_model_in[0].shape[0])
    min_arr_expanded=numpy.array([min_arr]*test_model_in[0].shape[0])
    all_pred=model.predict(test_model_in)
    pred=all_pred*bias_arr+min_arr_expanded
    gt=(test_model_out[0]*bias_arr+min_arr_expanded)
    mae_tmp=numpy.absolute(pred-gt)
    mape_tmp=mae_tmp/gt
    return mape_tmp,gt,pred
def fix_seeds(seed):#固定模型参数的初始值
    random.seed(seed)
    numpy.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    # K.set_session(sess)
from argparse import ArgumentParser
def get_arguments():
    # TODO 将聚类函数整合到此函数里，从而不需要对分段建模对象进行硬编码
    # 这里是直接采用事先聚类的结果
    argparser = ArgumentParser(description='build and train models')
    argparser.add_argument("--method", "-m", help="clustering algorithm: sfhc,k-means(km),spec(spectural),hc(hierarchical),hcc(hierarchical clustering by number of final clusters)")
    argparser.add_argument("--threshold", "-t", help="similarity threshold for clustering")
    argparser.add_argument("--name", "-n", help="model's name:stdn,mlp,cnn,dpf",default='mlp')
    results = argparser.parse_args()
    method=results.method
    threshold=results.threshold
    name=results.name
    if method=='spec':
        if threshold=='65':
            clusters=[[139, 37], [316], [46, 47], [138, 385, 41, 44], [33, 45], [34, 246, 42, 152], [43]]
        elif threshold=='70':
            clusters=[[139, 37], [316, 34], [46, 47], [138, 33, 45], [246], [385], [42, 41, 44], [43, 152]]
    elif method=='hc':
        if threshold=='70':
            clusters=[[33], [34, 42, 45, 152, 44, 46, 47, 316, 37, 43], [41, 139], [138, 385], [246]]
        elif threshold=='65':
            clusters=[[33], [34, 42, 45, 152, 44, 46, 47], [37, 43], [41, 139], [138, 316], [246], [385]]#在dataset 3中该聚类结果应该对应HCC
        elif threshold=='60':
            clusters=[[41, 54, 139, 153, 43], [42, 45, 46, 47, 59, 55, 56, 154, 57, 316], [44, 58], [246]]
    elif method=='hcc':
        if threshold=='70':
            clusters=[[33], [34, 42, 45, 152], [37, 43], [41, 139], [44, 46, 47], [138, 385], [246], [316]]
        elif threshold=='65':
            clusters=[[33], [34, 42, 45, 152, 44, 46, 47, 37, 43], [41, 139], [138, 316, 385], [246]]
    elif method=='km':
        if threshold=='70':
            clusters=[[139, 41], [316, 45], [46, 47, 138, 44], [33], [34, 42, 152], [246], [385], [37, 43]]
        elif threshold=='65':
            clusters=[[139, 33], [316, 45], [46, 385, 44], [47, 34, 42, 152], [138], [246], [37, 41, 43]]
    elif method=='sfhc':
        if threshold=='60':
            clusters=[[139, 46, 37, 41, 43, 44, 152], [316, 47, 138, 45, 385], [33], [34], [246], [42]]
        elif threshold=='70':
            clusters=[[139], [316, 46, 47, 138, 45, 44], [33], [34], [246], [385], [42], [37, 41, 43, 152]]
        elif threshold=='65':
            clusters=[[139], [316, 46, 47, 138, 34, 45, 44], [33], [246], [385], [42], [37, 41, 43, 152]]

        elif threshold=='50':
            clusters=[[153, 54, 56, 58, 316, 47, 45, 42, 55, 154, 41, 43, 44, 46, 139, 59, 57], [246]]

    elif method=='single':
        clusters=[[139], [316], [46], [47], [138], [33], [34], [45], [246], [385], [42], [37], [41], [43], [44], [152]]
    else:
        clusters=[[139,316,46,47,138,33,34,45,246,385,42,37,41,43,44,152]]
    return method+'_4',threshold,clusters,name
#各路段对应的最大最小旅行时间
seg_max_min_stdn={
    139: [166.29943739485032, 35.893463801719534],
    316: [227.2382333104532, 36.0],
    46: [132.49942189836304, 38.0],
    47: [146.13889161932389, 38.0],
    138: [209.8926384854106, 64.93723390249427],
    33: [286.89756780462835, 25.0],
    34: [152.99973579978757, 25.0],
    45: [280.4258384923905, 23.0],
    246: [336.7594258701566, 23.0],
    385: [164.1657527220043, 68.0],
    42: [142.53401090779946, 34.0],
    37: [385.8589364114241, 41.0],
    41: [273.8762736423545, 34.0],
    43: [144.94232522674483, 23.0],
    44: [74.57137580851355, 23.0],
    152: [394.2002417282169, 68.0]}
seg_max_min={
    139: [166.29943739485032, 35.893463801719534],
    316: [227.2382333104532, 36.0],
    46: [132.49942189836304, 38.0],
    47: [146.13889161932389, 38.0],
    138: [209.8926384854106, 64.93723390249427],
    33: [286.89756780462835, 25.0],
    34: [152.99973579978757, 25.0],
    45: [280.4258384923905, 23.0],
    246: [336.7594258701566, 23.0],
    385: [164.1657527220043, 68.0],
    42: [142.53401090779946, 34.0],
    37: [385.8589364114241, 41.0],
    41: [273.8762736423545, 34.0],
    43: [144.94232522674483, 23.0],
    44: [74.57137580851355, 23.0],
    152: [394.2002417282169, 68.0]}

