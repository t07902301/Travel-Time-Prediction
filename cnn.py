from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Dense, Activation, concatenate, Input, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D
from keras.optimizers import *
from utils.basic_functions import *
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
import pickle
test=numpy.load('/root/yiwei_2021_05/data/test_3.npy',allow_pickle=True).item()
train=numpy.load('/root/yiwei_2021_05/data/train_3.npy',allow_pickle=True).item()
train_days,train_dim=train[43].shape
test_days,test_dim=test[43].shape
look_back_span=4
def create_model(train_in_shape,train_out_shape,kernel_size):
    input_shape=(train_in_shape[1],train_in_shape[2],1)
    # print(input_shape)
    model = Sequential()
    model.add(Conv2D(3, kernel_size=(kernel_size,3),activation='relu',input_shape=input_shape,name='convd_1'))
    # model.add(Conv2D(8, kernel_size=(kernel_size,3),activation='relu',name='convd_2'))
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(24, activation='relu'))
    # model.add(Dropout(0.5))
#     model.add(Dense(24, activation='relu'))
#     model.add(Dropout(0.5))
    model.add(Dense(train_out_shape[1], activation='linear'))
    return model
# def create_model(train_in,train_out,kernel_size):
#     input_shape=Input(train_in.shape[2],train_in.shape[1],1)
#     convd_1=Conv2D(8, kernel_size=(kernel_size,3),activation='relu',input_shape=input_shape,name='convd_1')(input_shape)
#     flat_1=Flatten(name='flat_1')(convd_1)
#     dense_1=Dense(32,activation='relu',name='dense')(flat_1)
#     output_layer=Dense(train_out.shape[1], activation='linear',name='ouyput_layer')(dense_1)
#     model=Model(inputs=[input_shape],outputs=output_layer,name='cnn')
#     return model
def get_input(targets):
    
    train_flat=numpy.array([numpy.ravel(train[seg]) for seg in targets])
    # print(train_flat.shape)
    train_time_steps=numpy.array([train_flat[:,time_step] for time_step in range(train_flat.shape[1])])
    # print(train_time_steps.shape)
    # exit(0)
    test_flat=numpy.array([numpy.ravel(test[seg]) for seg in targets])
    test_time_steps=numpy.array([test_flat[:,time_step] for time_step in range(test_flat.shape[1])])
    train_in=[]
    train_out=[]
    for day in range(train_days):
        day_base=train_time_steps[day*train_dim:(day+1)*train_dim]
        for i in range(train_dim-look_back_span):
            train_in.append(numpy.transpose(day_base[i:i+look_back_span]))
            train_out.append(numpy.transpose(day_base[i+look_back_span])) 
            # train_in.append(day_base[i:i+look_back_span])
            # train_out.append(day_base[i+look_back_span]) 
    test_in=[]
    test_out=[]
    for day in range(test_days):
        day_base=test_time_steps[day*test_dim:(day+1)*test_dim]
        for i in range(test_dim-look_back_span):
            test_in.append(numpy.transpose(day_base[i:i+look_back_span]))
            test_out.append(numpy.transpose(day_base[i+look_back_span])) 
            # test_in.append(day_base[i:i+look_back_span])
            # test_out.append(day_base[i+look_back_span])           
    train_in=numpy.array(train_in)
    train_out=numpy.array(train_out)
    test_in=numpy.array(test_in)
    test_out=numpy.array(test_out)
    test_in=test_in.reshape(test_in.shape[0],test_in.shape[1],test_in.shape[2],1)
    train_in=train_in.reshape(train_in.shape[0],train_in.shape[1],train_in.shape[2],1)
    return (train_in,test_in,train_out,test_out)

def get_input_1(targets):
    train_in=[]
    train_out=[]
    test_in=[]
    test_out=[]
    train_flat=numpy.array([numpy.ravel(train[seg]) for seg in targets])
    test_flat=numpy.array([numpy.ravel(test[seg]) for seg in targets])
    train_flat=numpy.array(train_flat)
    test_flat=numpy.array(test_flat)
    train_flat_len=len(train_flat[0])
    test_flat_len=len(test_flat[0])
    for i in range(train_flat_len-look_back_span):
        train_in.append(train_flat[:,i:i+look_back_span])
        train_out.append(train_flat[:,i+look_back_span])
    for i in range(test_flat_len-look_back_span):
        test_in.append(test_flat[:,i:i+look_back_span])
        test_out.append(test_flat[:,i+look_back_span])
    train_in=numpy.array(train_in)
    train_out=numpy.array(train_out)
    test_in=numpy.array(test_in)
    test_out=numpy.array(test_out)
    test_in=test_in.reshape(test_in.shape[0],test_in.shape[1],test_in.shape[2],1)
    train_in=train_in.reshape(train_in.shape[0],train_in.shape[1],train_in.shape[2],1)
    return (train_in,test_in,train_out,test_out)
def train_and_test_simple(inputs,targets,history_path,model_path):
    train_in,test_in,train_out,test_out=inputs
    print(test_out.shape,train_out.shape,test_in.shape,train_in.shape)
    # print(test_dim)
    # exit(0)
    start=(train_dim-look_back_span)*(train_days-test_days)

    validate_in=train_in[start:]
    validate_out=train_out[start:]
    train_in_new=train_in[:start]
    train_out_new=train_out[:start]
    
    model_in=[train_in_new]
    model_out=[train_out_new]
    test_model_in=[test_in]
    test_model_out=[test_out]
    validate_in=[validate_in]
    validate_out=[validate_out]
    if len(targets)<3:
        kernel_size=len(targets)
    else:
        kernel_size=3
    train_in_shape=train_in_new.shape
    train_out_shape=train_out_new.shape
    fix_seeds(0)
    model=create_model(train_in_shape,train_out_shape,kernel_size)
    model.summary()
    model.compile(loss='mse', optimizer = 'adam') 
    # print('Done!')
    # exit(0)
    train_history=model.fit(x=model_in,y=model_out,batch_size=10,epochs=100,validation_data=(validate_in,validate_out),callbacks=[early_stopping],verbose=0) 
    with open(history_path, 'wb') as file_pi:
        pickle.dump(train_history.history,file_pi)
    # show_train_history(train_history,'loss','val_loss')
    model.save(model_path)
    mape_tmp,gt,pred=evaluate(model,targets,test_model_in,test_model_out)
    return mape_tmp

def main():
    method,threshold,clusters,name=get_arguments()
    
    mark='cnn_{}_{}_4'.format(method,threshold)
    mape=[]
    for index,c in enumerate(clusters):
        print('---Cluster {}-----'.format(index))
        print(c)
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