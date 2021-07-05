import pickle
# from tensorflow.keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Dense, Activation, concatenate, Input
from keras.models import Model
from keras.optimizers import *
from utils.basic_functions import *
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
test=numpy.load('/root/yiwei_2021_05/data/test_3.npy',allow_pickle=True).item()
train=numpy.load('/root/yiwei_2021_05/data/train_3.npy',allow_pickle=True).item()
train_days,train_dim=train[43].shape
test_days,test_dim=test[43].shape
look_back_span=6

def get_input(targets):
    #训练集
    X={}
    for seg in targets:
        record=train[seg]
        input_c=[]
        output=[]
        for day in range(train_days):
            valid=record[day] # a list
            for i in range(train_dim-look_back_span):
                input_c.append(valid[i:i+look_back_span])#前6个时间步
                output.append([valid[i+look_back_span]])
        X[seg]=[input_c,output]
    #连接
    input_x,output_x=X[targets[0]]
    input_x=numpy.array(input_x)
    output_x=numpy.array(output_x)
    for seg in targets[1:]:
        input_c,output=X[seg]
        input_x=numpy.concatenate((input_x,input_c),axis=1)
        output_x=numpy.concatenate((output_x,output),axis=1)
    #测试集
    Y={}
    for seg in targets:
        record=test[seg]
        input_c=[]
        output=[]
        for day in range(test_days):
            valid=record[day] # a list
            for i in range(test_dim-look_back_span):
                input_c.append(valid[i:i+look_back_span])
                output.append([valid[i+look_back_span]])
        Y[seg]=[input_c,output]
    #连接
    input_y,output_y=Y[targets[0]]
    input_y=numpy.array(input_y)
    output_y=numpy.array(output_y)
    for seg in targets[1:]:
        input_c,output=Y[seg]
        input_y=numpy.concatenate((input_y,input_c),axis=1)
        output_y=numpy.concatenate((output_y,output),axis=1)
    return (input_x,input_y,output_x,output_y)

def create_model(model_in_shape,model_out_shape):
    input_1=Input(shape=(model_in_shape[1],))
    # dense_1=Dense(32, use_bias = True,activation='linear',name='layer_1')(input_1)
    # dense_2=Dense(24, use_bias = True,activation='linear',name='layer_2')(dense_1)
    # dense_3=Dense(model_out_shape[1], use_bias = True,activation='linear')(dense_2)
    dense_3=Dense(model_out_shape[1], use_bias = True,activation='linear')(input_1)
    model=Model(inputs=[input_1],outputs=dense_3)
    return model
def train_and_test_simple(data,targets,history_path,model_path):

    train_in,test_in,train_out,test_out=data
    start=(train_dim-look_back_span)*(train_days-test_days)
    validate_in=train_in[start:]
    validate_out=train_out[start:]
    train_in_new=train_in[:start]
    train_out_new=train_out[:start]
    model_in=[train_in_new]
    model_out=[train_out_new]
    test_model_in=[test_in]
    test_model_out=[test_out]
    validate_model_in=[validate_in]
    validate_model_out=[validate_out]

    train_in_shape=train_in_new.shape
    train_out_shape=train_out.shape
    fix_seeds(0)
    model=create_model(train_in_shape,train_out_shape)
    model.summary()
    model.compile(loss='mse', optimizer = 'adam') 
    # model.compile(loss='mape', optimizer = 'adam') 

    # print(model_in[0].shape,model_out[0].shape)
    # print(train_in.shape,train_out.shape)
    # exit(0)
    train_history=model.fit(x=model_in,y=model_out,batch_size=10,epochs=100,validation_data=(validate_model_in,validate_model_out),callbacks=[early_stopping],verbose=0)
    with open(history_path, 'wb') as file_pi:
        pickle.dump(train_history.history,file_pi)
    #cluster中元素对应的index
    max_arr=numpy.array([ seg_max_min[seg][0] for seg in targets])
    min_arr=numpy.array([seg_max_min[seg][1] for seg in targets])
    bias_arr=max_arr-min_arr#(路段0，路段1,...)
    min_arr_expanded=numpy.array([min_arr]*test_in.shape[0])
    all_pred=model.predict(test_model_in)
    pred=all_pred*bias_arr+min_arr_expanded
    gt=(test_model_out[0]*bias_arr+min_arr_expanded)
    mae_tmp=numpy.absolute(pred-gt)
    mape_tmp=mae_tmp/gt
    model.save(model_path)
    return mape_tmp

def main():
    method,threshold,clusters,name=get_arguments()
    mark='mlp_{}_{}_none'.format(method,threshold)
    mape=[]
    for index,c in enumerate(clusters):
        print('---Cluster {}-----'.format(index))
        print(c)
        # train_in_c,test_in_c,train_out_c,test_out_c,train_in_d,test_in_d=get_input(c)
        data=get_input(c)
        history_path='/root/yiwei_2021_05/data/history/{}_{}.txt'.format(mark,index)
        model_path='models/{}_{}.h5'.format(mark,index)
        result_stdn=train_and_test_simple(data,c,history_path,model_path)
        mape.append(result_stdn)
    mape_result=mape[0]
    for i in range(1,len(mape)):
        mape_result=numpy.concatenate((mape_result,mape[i]),axis=1)
    with open('result/{}.txt'.format(mark),'w') as f:
        f.write('{}:{}%\n'.format(mape_result.shape,numpy.around(numpy.mean(mape_result)*100,2)))
if __name__ == "__main__":
    main()
