from os import name
import numpy
from utils.basic_functions import *
from utils.modules import create_stdn
from keras.models import load_model
test=numpy.load('/root/yiwei_2021_05/data/test.npy',allow_pickle=True).item()
dw=numpy.load('/root/yiwei_2021_05/data/day_of_week.npy',allow_pickle=True).item()
ts=numpy.load('/root/yiwei_2021_05/data/encoded_time_stamp.npy')
test_days,test_dim=test[43].shape
look_back_span=4
def get_cnn_input(targets):
    test_flat=numpy.array([numpy.ravel(test[seg]) for seg in targets])
    test_time_steps=numpy.array([test_flat[:,time_step] for time_step in range(test_flat.shape[1])])
    test_in=[]
    test_out=[]
    for day in range(test_days):
        day_base=test_time_steps[day*test_dim:(day+1)*test_dim]
        for i in range(test_dim-look_back_span):
            test_in.append(numpy.transpose(day_base[i:i+look_back_span]))
            test_out.append(numpy.transpose(day_base[i+look_back_span])) 
            # test_in.append(day_base[i:i+look_back_span])
            # test_out.append(day_base[i+look_back_span])           
    test_in=numpy.array(test_in)
    test_out=numpy.array(test_out)
    test_in=test_in.reshape(test_in.shape[0],test_in.shape[1],test_in.shape[2],1)
    return (test_in,test_out)
def evaluate_cnn(inputs,targets,model_path):
    test_in,test_out=inputs
    test_model_in=[test_in]
    test_model_out=[test_out]
    model=load_model(model_path)
    mape_tmp,gt,pred=evaluate(model,targets,test_model_in,test_model_out)
    return mape_tmp,gt,pred
def get_dpf_input(targets,look_back_days,test_dates):
    test_flat=numpy.array([numpy.ravel(test[seg]) for seg in targets])
    test_time_steps=numpy.array([test_flat[:,time_step] for time_step in range(test_flat.shape[1])])
    test_in_c=[]
    test_out_c=[]
    test_in_d=[]
    for day in range(look_back_days,test_days):
        day_base=test_time_steps[day*test_dim:(day+1)*test_dim]
        for i in range(test_dim-look_back_span):
            test_in_c.append(day_base[i:i+look_back_span])
            test_out_c.append(day_base[i+look_back_span]) 
            test_in_d.append([test_time_steps[(day-i)*test_dim+i] for i in range(look_back_days)])
    test_in_c=numpy.array(test_in_c)
    test_out_c=numpy.array(test_out_c)
    test_in_d=numpy.array(test_in_d)

    #加入日期和时间点(每个都具有输出值)

    dw_test=[]
    for i in test_dates:
        dw_test+=[dw[i]]*(test_dim-look_back_span)

    ts_test=[]
    for i in range(look_back_days,test_days):
        ts_test=numpy.concatenate((ts_test,ts[:test_dim-look_back_span]))

    ts_test=ts_test.reshape(ts_test.shape[0],1)
    
    dw_test=numpy.array(dw_test)
    dw_test=dw_test.reshape(dw_test.shape[0],1)
    return (test_in_c,test_out_c,test_in_d,ts_test,dw_test)
def evaluate_dpf(inputs,targets,model_path):
    test_in_c,test_out,test_in_d,ts_test,dw_test=inputs
    test_model_in=[test_in_c,test_in_d,dw_test,ts_test]
    test_model_out=[test_out] 
    model=load_model(model_path)   
    mape_tmp,gt,pred=evaluate(model,targets,test_model_in,test_model_out)
    return mape_tmp,gt,pred
def get_mlp_input(targets):
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
    return (input_y,output_y)
def evaluate_mlp(data,targets,model_path):
    test_in,test_out=data
    test_model_in=[test_in]
    test_model_out=[test_out]
    model=load_model(model_path)
    max_arr=numpy.array([ seg_max_min[seg][0] for seg in targets])
    min_arr=numpy.array([seg_max_min[seg][1] for seg in targets])
    bias_arr=max_arr-min_arr#(路段0，路段1,...)
    min_arr_expanded=numpy.array([min_arr]*test_in.shape[0])
    all_pred=model.predict(test_model_in)
    pred=all_pred*bias_arr+min_arr_expanded
    gt=(test_model_out[0]*bias_arr+min_arr_expanded)
    mae_tmp=numpy.absolute(pred-gt)
    mape_tmp=mae_tmp/gt
    return mape_tmp,gt,pred
def get_stdn_input(targets,test,look_back_days,shift,test_days,test_dim):
    test_flat=numpy.array([numpy.ravel(test[seg]) for seg in targets])
    test_time_steps=numpy.array([test_flat[:,time_step] for time_step in range(test_flat.shape[1])])
    test_in_c=[]
    test_out_c=[]
    test_in_d=[]
    for day in range(look_back_days,test_days):
        day_base=test_time_steps[day*test_dim:(day+1)*test_dim]
        previous_days=[test_time_steps[(day-j)*test_dim:(day-j+1)*test_dim] for j in range(look_back_days)]
        dim_per_day=test_dim-look_back_span
        shift_span=(shift-1)//2
        for i in range(shift_span,dim_per_day-shift_span):
            test_in_c.append(day_base[i:i+look_back_span])
            test_out_c.append(day_base[i+look_back_span]) 
            test_in_d.append([previous_days[j][i-shift_span:i+shift_span+1] for j in range(look_back_days)])
    test_in_c=numpy.array(test_in_c)
    test_out_c=numpy.array(test_out_c)
    test_in_d=numpy.array(test_in_d)
    return (test_in_c,test_out_c,test_in_d)

def evaluate_stdn_mock(data,targets,model_path,look_back_days,shift,hidden_units):
    test_in_c,test_out,test_in_d=data
    test_in_d_split=[test_in_d[:,j,:,:] for j in range(look_back_days)]    
    test_model_in=[test_in_c]+test_in_d_split
    test_model_out=[test_out]
    model=create_stdn(look_back_span,shift,len(targets),look_back_days,hidden_units)
    model.load_weights(model_path)
    # model.load_weights('models/stdn_weights.h5',by_name=True)
    mape_tmp,gt,pred=evaluate_stdn(model,targets,test_model_in,test_model_out)
    return mape_tmp,gt,pred
def evaluate_convlstm(data,targets,model_path):
    test_in,test_out=data
    test_in=test_in.reshape(test_in.shape[0],test_in.shape[1],test_in.shape[2],1,1)
    test_out=test_out.reshape(test_out.shape[0],1,test_out.shape[1],1,1)
    test_model_in=[test_in]
    test_model_out=[test_out]
    model=load_model(model_path)
    result=convlstm(model,targets,test_model_in,test_model_out)
    return result
def convlstm(model,targets,test_model_in,test_model_out):
    max_arr=numpy.array([ seg_max_min[seg][0] for seg in targets])
    min_arr=numpy.array([seg_max_min[seg][1] for seg in targets])
    bias_arr=max_arr-min_arr#(路段0，路段1,...)
    print(min_arr.shape)
    bias_arr=bias_arr.reshape(1,len(targets),1,1)
    min_arr=min_arr.reshape(1,len(targets),1,1)
    min_arr_expanded=numpy.array([min_arr]*test_model_in[0].shape[0])

    gt=(test_model_out[0]*bias_arr+min_arr_expanded)
    all_pred=model.predict(test_model_in)
    pred=all_pred*bias_arr+min_arr_expanded
    mae_tmp=numpy.absolute(pred-gt)
    mape_tmp=mae_tmp/gt
    return mape_tmp
def get_convlstm_input(targets):
    test_flat=numpy.array([numpy.ravel(test[seg]) for seg in targets])
    test_time_steps=numpy.array([test_flat[:,time_step] for time_step in range(test_flat.shape[1])])
    test_in=[]
    test_out=[]
    for day in range(test_days):
        day_base=test_time_steps[day*test_dim:(day+1)*test_dim]
        for i in range(test_dim-look_back_span):
            test_in.append(day_base[i:i+look_back_span])
            test_out.append(day_base[i+look_back_span]) 
    test_in=numpy.array(test_in)
    test_out=numpy.array(test_out)
    return test_in,test_out

def main():
    gt_value={}
    pred_value={}
    ape_dict={}
    method,threshold,clusters,name=get_arguments()
    mark='{}_{}_{}'.format(name,method,threshold)
    if name=='stdn':
        hidden_units=8
        mark='{}_{}_{}_{}'.format(name,method,threshold,hidden_units)     
    # name='stdn'
    # hidden_units=8
    # mark='{}_{}_{}_{}'.format(name,method,threshold,hidden_units) 
    # name='dpf'
    # mark='{}_{}_{}_embed_1_4'.format(name,method,threshold)
    # mark='{}_{}_{}_stacked'.format(name,method,threshold)
    # mark='{}_{}_{}'.format(name,method,threshold)    
    # name='mlp'
    # mark='{}_{}_{}_linear_init'.format(name,method,threshold)
    # mark='{}_{}_{}'.format(name,method,threshold)
    # name='cnn'
    # # mark='{}_{}_{}_3_24'.format(name,method,threshold)
    # mark='{}_{}_{}'.format(name,method,threshold)
    # # mark='{}_{}_{}_4'.format(name,method,threshold)

    mape=[]
    for index,c in enumerate(clusters):
        model_path='models/mlp_models/{}_{}.h5'.format(mark,index)
        print(model_path)
        if name=='cnn':
            data=get_cnn_input(c)
            result,gt,pred=evaluate_cnn(data,c,model_path)
            mape.append(result)
            for idx,seg in enumerate(c):
                ape_dict[seg]=result[:,idx]
                pred_value[seg]=pred[:,idx]
                gt_value[seg]=gt[:,idx]
        elif name=='dpf':
            look_back_days=0
            dates=['0503', '0504', '0505', '0508', '0509', '0510', '0511', '0512', '0515', '0516', '0517', '0518', '0519', '0522', '0523', '0524', '0525', '0526', '0529', '0530', '0531', '0602', '0605', '0606', '0607', '0608', '0609', '0612', '0613', '0614', '0615', '0616', '0620', '0621', '0622', '0623', '0626', '0627', '0628', '0629', '0630']
            test_days_dates=dates[33:]
            data=get_dpf_input(c,look_back_days,test_days_dates)
            result,gt,pred=evaluate_dpf(data,c,model_path)
            mape.append(result) 
            for idx,seg in enumerate(c):
                ape_dict[seg]=result[:,idx]
                pred_value[seg]=pred[:,idx]
                gt_value[seg]=gt[:,idx]
        elif 'mlp' in name:
            data=get_mlp_input(c)
            result,gt,pred=evaluate_mlp(data,c,model_path)
            mape.append(result)
            for idx,seg in enumerate(c):
                ape_dict[seg]=result[:,idx]
                pred_value[seg]=pred[:,idx]
                gt_value[seg]=gt[:,idx]

        elif name=='stdn':
            look_back_days=3
            shift=13
            model_path='models/{}_{}_weights.h5'.format(mark,index)
            print(model_path)
            test=numpy.load('/root/yiwei_2021_05/data/test_stdn.npy',allow_pickle=True).item()
            test_days,test_dim=test[43].shape
            data=get_stdn_input(c,test,look_back_days,shift,test_days,test_dim)
            result,gt,pred=evaluate_stdn_mock(data,c,model_path,look_back_days,shift,hidden_units)
            mape.append(result) 
            for idx,seg in enumerate(c):
                ape_dict[seg]=result[:,idx]
                pred_value[seg]=pred[:,idx]
                gt_value[seg]=gt[:,idx]            
        elif name=='convlstm':
            data=get_convlstm_input(c)
            result=evaluate_convlstm(data,c,model_path)
            mape.append(result)            
    mape_result=mape[0]
    for i in range(1,len(mape)):
        mape_result=numpy.concatenate((mape_result,mape[i]),axis=1)
    # check_peak(mape_result,name)
    # print(check_jam(ape_dict))
    print(mape_result.shape,numpy.around(numpy.mean(mape_result)*100,2))
    #TODO 可视化比较预测值和实际值的差异，可以将是否进行可视化的选择加入参数设置中
    # numpy.save('result/error/{}_mape.npy'.format(mark),ape_dict)
    # numpy.save('result/error/{}_pred.npy'.format(mark),pred_value)
    # numpy.save('result/error/{}_gt.npy'.format(mark),gt_value)
    print(mark)
jam=[46,33,34,246,42,41,43,152]
def check_jam(ape_dict):#拥堵时段的预测情况
    jam_ape=ape_dict[jam[0]]
    for i in range(1,len(jam)):
        jam_ape=numpy.concatenate((jam_ape,ape_dict[jam[i]]))
    jam_ape=numpy.array([ape_dict[seg] for seg in jam])
    # jam_mape=numpy.around(numpy.mean(jam_ape,axis=1),4)*100
    jam_mape=numpy.around(numpy.mean(jam_ape),4)*100
    return jam_ape.shape,jam_mape
def check_peak(pred,model_name):#高峰时段的预测情况
    if model_name=='stdn':
        points=[i for i in range(1113)]
        peak_hours=[points[159*i+6:159*i+18]+points[159*i+108:159*i+132] for i in range(7)]
        # peak_hours=[points[159*i+108:159*i+132] for i in range(7)]
    else:
        points=[i for i in range(1368)]
        peak_hours=[points[171*i+6:171*i+18]+points[171*i+108:171*i+144] for i in range(8)]

        # peak_hours=[points[171*i+108:171*i+132] for i in range(8)]
        # peak_hours=[points[171*i+6:171*i+18] for i in range(8)]

    peak_hours=numpy.concatenate(peak_hours,axis=0)
    pred_peak=pred[peak_hours]
    # pred_peak=pred[selected,:]
    print(pred_peak.shape,numpy.around(numpy.mean(pred_peak)*100,2))
    # if 'single' in method:
    #     numpy.save('result/mlp_auto_regression.npy',auto)
    # else:
    #     numpy.save('result/mlp_neighor.npy',neighbor)
if __name__ == "__main__":
    main()