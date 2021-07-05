from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# from tensorflow.keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Dense, Activation, concatenate, Input, Reshape, Flatten, Dropout, LSTM,GRU
# from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import *
# import matplotlib.pyplot as plot
from utils.basic_functions import *
import pickle
dw=numpy.load('/root/yiwei_2021_05/data/day_of_week.npy',allow_pickle=True).item()
ts=numpy.load('/root/yiwei_2021_05/data/encoded_time_stamp.npy')
test=numpy.load('/root/yiwei_2021_05/data/test_3.npy',allow_pickle=True).item()
train=numpy.load('/root/yiwei_2021_05/data/train_3.npy',allow_pickle=True).item()
train_days,train_dim=train[43].shape
test_days,test_dim=test[43].shape
look_back_span=4
def get_input(targets,look_back_days,train_dates,test_dates):
    train_flat=numpy.array([numpy.ravel(train[seg]) for seg in targets])
    train_time_steps=numpy.array([train_flat[:,time_step] for time_step in range(train_flat.shape[1])])
    test_flat=numpy.array([numpy.ravel(test[seg]) for seg in targets])
    test_time_steps=numpy.array([test_flat[:,time_step] for time_step in range(test_flat.shape[1])])
    train_in_c=[]
    train_out_c=[]
    train_in_d=[]
    for day in range(look_back_days,train_days):
        day_base=train_time_steps[day*train_dim:(day+1)*train_dim]
        for i in range(train_dim-look_back_span):
            train_in_c.append(day_base[i:i+look_back_span])
            train_out_c.append(day_base[i+look_back_span]) 
            train_in_d.append([train_time_steps[(day-i)*train_dim+i] for i in range(look_back_days)])
    test_in_c=[]
    test_out_c=[]
    test_in_d=[]
    for day in range(look_back_days,test_days):
        day_base=test_time_steps[day*test_dim:(day+1)*test_dim]
        for i in range(test_dim-look_back_span):
            test_in_c.append(day_base[i:i+look_back_span])
            test_out_c.append(day_base[i+look_back_span]) 
            test_in_d.append([test_time_steps[(day-i)*train_dim+i] for i in range(look_back_days)])
    train_in_c=numpy.array(train_in_c)
    train_out_c=numpy.array(train_out_c)
    test_in_c=numpy.array(test_in_c)
    test_out_c=numpy.array(test_out_c)
    train_in_d=numpy.array(train_in_d)
    test_in_d=numpy.array(test_in_d)

    #加入日期和时间点(每个都具有输出值)
    dw_train=[]
    for i in train_dates:
        dw_train+=[dw[i]]*(train_dim-look_back_span) 

    dw_test=[]
    for i in test_dates:
        dw_test+=[dw[i]]*(test_dim-look_back_span)

    ts_train=[]
    for i in range(look_back_days,train_days):
        ts_train=numpy.concatenate((ts_train,ts[:train_dim-look_back_span]))
    ts_test=[]
    for i in range(look_back_days,test_days):
        ts_test=numpy.concatenate((ts_test,ts[:test_dim-look_back_span]))

    ts_train=ts_train.reshape(ts_train.shape[0],1)
    ts_test=ts_test.reshape(ts_test.shape[0],1)
    
    dw_test=numpy.array(dw_test)
    dw_train=numpy.array(dw_train)
    dw_test=dw_test.reshape(dw_test.shape[0],1)
    dw_train=dw_train.reshape(dw_train.shape[0],1)
    return (train_in_c,test_in_c,train_out_c,test_out_c,train_in_d,test_in_d,ts_train,ts_test,dw_test,dw_train)

from keras.layers import Embedding 
def create_nde(model_in,model_out):
    train_in_c,train_in_d=model_in
    input_nearest=Input(shape=train_in_c.shape[1:],name='nearest_inputs')
    input_day=Input(shape=train_in_d.shape[1:],name='daily_inputs')
    dw_input=Input(shape=(1,),name='day_of_week_inputs')
    ts_input=Input(shape=(1,),name='time_of_day_inputs')

    #nearest
    temporal_n_1=GRU(32, use_bias = True,return_sequences=True,name='nearest_1')(input_nearest)
    temporal_n_2=GRU(32, use_bias = True,return_sequences=True,name='nearest_2')(temporal_n_1)
    temporal_n_3=GRU(16, use_bias = True,return_sequences=False,name='nearest_3')(temporal_n_2)

    #nearest
    temporal_d_1=GRU(32, use_bias = True,return_sequences=True,name='daily_1')(input_nearest)
    temporal_d_2=GRU(32, use_bias = True,return_sequences=True,name='daily_2')(temporal_d_1)
    temporal_d_3=GRU(16, use_bias = True,return_sequences=False,name='daily_3')(temporal_d_2)
    temporal_layer=concatenate([temporal_n_3,temporal_d_3])
    # temporal_drop=Dropout(0.5)(temporal_layer)
    
    # time_embedded=Embedding(180,3,input_length=1,name='time_of_day')(ts_input)
    # flat_1=Flatten(name='flat_1')(time_embedded)
    # day_embedded=Embedding(7,3,input_length=1,name='day_of_week')(dw_input)
    # flat_2=Flatten(name='flat_2')(day_embedded)
    # external_factor=concatenate([flat_1,flat_2])

    # total=concatenate([temporal_layer,external_factor])
    # total_drop=Dropout(0.5)(total)

    fc_0=Dense(24, use_bias = True,activation='relu')(temporal_layer)
    fc_2=Dense(model_out.shape[1], use_bias = True,activation='linear',name='output_layer')(fc_0)

    model=Model(inputs=[input_nearest,input_day,dw_input,ts_input],outputs=fc_2,name='DL-NED')
    return model
def create_model(model_in,model_out):
    train_in_c_shape,train_in_d_shape=model_in
    input_nearest=Input(shape=train_in_c_shape[1:],name='nearest_inputs')
    input_day=Input(shape=train_in_d_shape[1:],name='daily_inputs')

    # train_in_c,train_in_d=model_in
    # input_nearest=Input(shape=train_in_c.shape[1:],name='nearest_inputs')
    # input_day=Input(shape=train_in_d.shape[1:],name='daily_inputs')
    dw_input=Input(shape=(1,),name='day_of_week_inputs')
    ts_input=Input(shape=(1,),name='time_of_day_inputs')

    #nearest
    # temporal_n_1=GRU(32, use_bias = True,return_sequences=True,name='nearest_1')(input_nearest)
    # temporal_n_2=GRU(32, use_bias = True,return_sequences=True,name='nearest_2')(temporal_n_1)
    # temporal_n_3=GRU(16, use_bias = True,return_sequences=False,name='nearest_3')(temporal_n_2)
    temporal_n_1=GRU(16, use_bias = True,return_sequences=False,name='nearest_1')(input_nearest)#simple gru
    
    # time_embedded=Embedding(180,3,input_length=1,name='time_of_day')(ts_input)
    # flat_1=Flatten(name='flat_1')(time_embedded)
    # day_embedded=Embedding(7,3,input_length=1,name='day_of_week')(dw_input)
    # flat_2=Flatten(name='flat_2')(day_embedded)
    # external_factor=concatenate([flat_1,flat_2])

    # total=concatenate([temporal_n_3,flat_1])
    # total_drop=Dropout(0.5)(total)
    # total=concatenate([temporal_n_1,external_factor])
    # total=concatenate([temporal_n_1,flat_1])


    # fc_0=Dense(16, use_bias = True,activation='relu')(total)
    # fc_2=Dense(model_out.shape[1], use_bias = True,activation='linear',name='output_layer')(fc_0)
    # fc_2=Dense(model_out.shape[1], use_bias = True,activation='linear',name='output_layer')(total)
    # fc_2=Dense(model_out.shape[1], use_bias = True,activation='linear',name='output_layer')(temporal_n_1)

    fc_2=Dense(model_out[1], use_bias = True,activation='linear',name='output_layer')(temporal_n_1)


    model=Model(inputs=[input_nearest,input_day,dw_input,ts_input],outputs=fc_2,name='DL-NE')
    return model

def train_and_test_simple(inputs,targets,history_path,model_path):
    train_in_c,test_in_c,train_out,test_out,train_in_d,test_in_d,ts_train,ts_test,dw_test,dw_train=inputs
    start=(train_dim-look_back_span)*(train_days-test_days)

    validate_in_c=train_in_c[start:]
    validate_in_d=train_in_d[start:]
    validate_ts=ts_train[start:]
    validate_dw=dw_train[start:]
    validate_out=train_out[start:]

    train_in_new_c=train_in_c[:start]
    train_in_new_d=train_in_d[:start]
    ts_train_new=ts_train[:start]
    dw_train_new=dw_train[:start]
    train_out_new=train_out[:start]

    model_in=[train_in_new_c,train_in_new_d,dw_train_new,ts_train_new]
    model_out=[train_out_new]
    test_model_in=[test_in_c,test_in_d,dw_test,ts_test]
    test_model_out=[test_out]
    validate_in=[validate_in_c,validate_in_d,validate_dw,validate_ts]
    validate_out=[validate_out]
    fix_seeds(0)
    input_shapes=(train_in_new_c.shape,train_in_new_d.shape)
    output_shapes=train_out_new.shape
    model=create_model(input_shapes,output_shapes)
    # model=create_model(model_in[:2],train_out_new)
    # model=create_nde(model_in[:2],train_out_new)
    model.summary()
    model.compile(loss='mse', optimizer = 'adam') 
    train_history=model.fit(x=model_in,y=model_out,batch_size=10,epochs=100,validation_data=(validate_in,validate_out),callbacks=[early_stopping],verbose=0) 
    with open(history_path, 'wb') as file_pi:
        pickle.dump(train_history.history,file_pi)
    model.save(model_path)
    mape_tmp,gt,pre=evaluate(model,targets,test_model_in,test_model_out)
    return mape_tmp

def main():
    method,threshold,clusters,name=get_arguments()
    mark='dpf_{}_{}_no_stacked'.format(method,threshold)
    look_back_days=0
    # dates=[ '0501', '0502', '0503', '0504', '0505', '0506', '0507','0508', '0509', '0510', '0511', '0512', '0513', '0514', '0515','0516', '0517', '0518', '0519', '0520', '0521', '0522', '0523','0524', '0525', '0526', '0527', '0528', '0529', '0530', '0531', '0602', '0603', '0604', '0605', '0606', '0607', '0608','0609', '0610', '0611', '0612', '0613', '0614', '0615', '0616','0617', '0618', '0619', '0620', '0621', '0622', '0623', '0624','0625', '0626', '0627', '0628', '0629', '0630']
    dates=['0503', '0504', '0505', '0508', '0509', '0510', '0511', '0512', '0515', '0516', '0517', '0518', '0519', '0522', '0523', '0524', '0525', '0526', '0529', '0530', '0531', '0602', '0605', '0606', '0607', '0608', '0609', '0612', '0613', '0614', '0615', '0616', '0620', '0621', '0622', '0623', '0626', '0627', '0628', '0629', '0630']
    train_days_dates=dates[look_back_days:train_days]
    test_days_dates=dates[train_days+look_back_days:]
    mape=[]
    for index,c in enumerate(clusters):
        print('---Cluster {}-----'.format(index))
        print(c)
        history_path='/root/yiwei_2021_05/data/history/{}_{}.txt'.format(mark,index)
        model_path='models/{}_{}.h5'.format(mark,index)
        input_data=get_input(c,look_back_days,train_days_dates,test_days_dates)
        result_ne=train_and_test_simple(input_data,c,history_path,model_path)
        mape.append(result_ne)
    mape_result=mape[0]
    for i in range(1,len(mape)):
        mape_result=numpy.concatenate((mape_result,mape[i]),axis=1)
    # mape_result.shape,numpy.mean(mape_result)
    with open('result/{}.txt'.format(mark),'w') as f:
        f.write('{}:{}%\n'.format(mape_result.shape,numpy.around(numpy.mean(mape_result)*100,2)))
if __name__ == "__main__":
    main()