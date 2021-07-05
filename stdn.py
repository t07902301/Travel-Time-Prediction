from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# from tensorflow.keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Dense, Activation, Input, Reshape, Dropout, BatchNormalization, Concatenate, LSTM,GRU
# from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
import keras.backend as K

import sys
sys.path.append('/root/yiwei_2021_05/data/scripts/utils/')
from utils.basic_functions import *
look_back_span=6
look_back_days=3
shift=13
test=numpy.load('/root/yiwei_2021_05/data/test_3_stdn.npy',allow_pickle=True).item()
train=numpy.load('/root/yiwei_2021_05/data/train_3_stdn.npy',allow_pickle=True).item()
train_days,train_dim=train[43].shape
test_days,test_dim=test[43].shape

def get_input(targets):
    train_flat=numpy.array([numpy.ravel(train[seg]) for seg in targets])
    train_time_steps=numpy.array([train_flat[:,time_step] for time_step in range(train_flat.shape[1])])
    test_flat=numpy.array([numpy.ravel(test[seg]) for seg in targets])
    test_time_steps=numpy.array([test_flat[:,time_step] for time_step in range(test_flat.shape[1])])
    train_in_c=[]
    train_out_c=[]
    train_in_d=[]
    for day in range(look_back_days,train_days):
        day_base=train_time_steps[day*train_dim:(day+1)*train_dim]
        previous_days=[train_time_steps[(day-j)*train_dim:(day-j+1)*train_dim] for j in range(look_back_days)]
        dim_per_day=train_dim-look_back_span
        shift_span=(shift-1)//2
        for i in range(shift_span,dim_per_day-shift_span):
            train_in_c.append(day_base[i:i+look_back_span])
            train_out_c.append(day_base[i+look_back_span])
            train_in_d.append([previous_days[j][i-shift_span:i+shift_span+1] for j in range(look_back_days)])
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

    train_in_c=numpy.array(train_in_c)
    train_out_c=numpy.array(train_out_c)
    test_in_c=numpy.array(test_in_c)
    test_out_c=numpy.array(test_out_c)
    train_in_d=numpy.array(train_in_d)
    test_in_d=numpy.array(test_in_d)
    return (train_in_c,test_in_c,train_out_c,test_out_c,train_in_d,test_in_d)

def create_model(time_steps_c,time_steps_d,dim,look_back_days,units_dim):#dim in current day and past days are same here
    current_input=Input(shape=(time_steps_c,dim),name='current_input')
    lstm = GRU(units=units_dim, return_sequences=False, dropout=0.1, recurrent_dropout=0.1,name='lstm')(current_input)
    
    day_inputs=[Input(shape=(time_steps_d,dim),name='day_input_{}'.format(i)) for i in range(look_back_days)]

    att_lstms = [GRU(units=units_dim, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{}".format(att + 1))(day_inputs[att]) for att in range(look_back_days)]
    #compare
    att_low_level=[Attention(method='cba')([att_lstms[att], lstm]) for att in range(look_back_days)]
    att_low_level=Concatenate(axis=-1)(att_low_level)
    att_low_level=Reshape(target_shape=(look_back_days, units_dim))(att_low_level)

    att_high_level = GRU(units=units_dim, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

    lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
    # lstm_all = Dropout(rate = 0.5)(lstm_all)
    lstm_all = Dense(units = dim)(lstm_all)
    pred_volume = Activation('linear')(lstm_all)
    model=Model(inputs=[current_input]+day_inputs,outputs=pred_volume)
    return model
from keras.layers import Layer
import keras.backend as K

class Attention(Layer):
    def __init__(self, method=None, **kwargs):
        self.supports_masking = True
        if method != 'lba' and method !='ga' and method != 'cba' and method is not None:
            raise ValueError('attention method is not supported')
        self.method = method
        super(Attention, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'method': self.method 
        })
        return config
    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.att_size = input_shape[0][-1]
            self.query_dim = input_shape[1][-1]
            if self.method == 'ga' or self.method == 'cba':
                self.Wq = self.add_weight(name='kernal_query_features', shape=(self.query_dim, self.att_size), initializer='glorot_normal', trainable=True)
        else:
            self.att_size = input_shape[-1]

        if self.method == 'cba':
            self.Wh = self.add_weight(name='kernal_hidden_features', shape=(self.att_size,self.att_size), initializer='glorot_normal', trainable=True)
        if self.method == 'lba' or self.method == 'cba':
            self.v = self.add_weight(name='query_vector', shape=(self.att_size, 1), initializer='zeros', trainable=True)

        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        '''
        :param inputs: a list of tensor of length not larger than 2, or a memory tensor of size BxTXD1.
        If a list, the first entry is memory, and the second one is query tensor of size BxD2 if any
        :param mask: the masking entry will be directly discarded
        :return: a tensor of size BxD1, weighted summing along the sequence dimension
        '''
        if isinstance(inputs, list) and len(inputs) == 2:
            memory, query = inputs
            if self.method is None:
                return memory[:,-1,:]
            elif self.method == 'cba':
                hidden = K.dot(memory, self.Wh) + K.expand_dims(K.dot(query, self.Wq), 1)
                hidden = K.tanh(hidden)
                s = K.squeeze(K.dot(hidden, self.v), -1)
            elif self.method == 'ga':
                s = K.sum(K.expand_dims(K.dot(query, self.Wq), 1) * memory, axis=-1)
            else:
                s = K.squeeze(K.dot(memory, self.v), -1)
            if mask is not None:
                mask = mask[0]
        else:
            if isinstance(inputs, list):
                if len(inputs) != 1:
                    raise ValueError('inputs length should not be larger than 2')
                memory = inputs[0]
            else:
                memory = inputs
            if self.method is None:
                return memory[:,-1,:]
            elif self.method == 'cba':
                hidden = K.dot(memory, self.Wh)
                hidden = K.tanh(hidden)
                s = K.squeeze(K.dot(hidden, self.v), -1)
            elif self.method == 'ga':
                raise ValueError('general attention needs the second input')
            else:
                s = K.squeeze(K.dot(memory, self.v), -1)

        s = K.softmax(s)
        if mask is not None:
            s *= K.cast(mask, dtype='float32')
            sum_by_time = K.sum(s, axis=-1, keepdims=True)
            s = s / (sum_by_time + K.epsilon())
        return K.sum(memory * K.expand_dims(s), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            att_size = input_shape[0][-1]
            batch = input_shape[0][0]
        else:
            att_size = input_shape[-1]
            batch = input_shape[0]
        return (batch, att_size)

class SimpleAttention(Layer):
    def __init__(self, method=None, **kwargs):
        self.supports_masking = True
        if method != 'lba' and method !='ga' and method != 'cba' and method is not None:
            raise ValueError('attention method is not supported')
        self.method = method
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.att_size = input_shape[0][-1]
            self.query_dim = input_shape[1][-1] + self.att_size
        else:
            self.att_size = input_shape[-1]
            self.query_dim = self.att_size

        if self.method == 'cba' or self.method == 'ga':
            self.Wq = self.add_weight(name='kernal_query_features', shape=(self.query_dim, self.att_size),
                                      initializer='glorot_normal', trainable=True)
        if self.method == 'cba':
            self.Wh = self.add_weight(name='kernal_hidden_features', shape=(self.att_size, self.att_size), initializer='glorot_normal', trainable=True)

        if self.method == 'lba' or self.method == 'cba':
            self.v = self.add_weight(name='query_vector', shape=(self.att_size, 1), initializer='zeros', trainable=True)

        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        '''
        :param inputs: a list of tensor of length not larger than 2, or a memory tensor of size BxTXD1.
        If a list, the first entry is memory, and the second one is query tensor of size BxD2 if any
        :param mask: the masking entry will be directly discarded
        :return: a tensor of size BxD1, weighted summing along the sequence dimension
        '''
        query = None
        if isinstance(inputs, list):
            memory = inputs[0]
            if len(inputs) > 1:
                query = inputs[1]
            elif len(inputs) > 2:
                raise ValueError('inputs length should not be larger than 2')
            if isinstance(mask, list):
                mask = mask[0]
        else:
            memory = inputs

        input_shape = K.int_shape(memory)
        if len(input_shape) >3:
            input_length = input_shape[1]
            memory = K.reshape(memory, (-1,) + input_shape[2:])
            if mask is not None:
                mask = K.reshape(mask, (-1,) + input_shape[2:-1])
            if query is not None:
                raise ValueError('query can be not supported')

        last = memory[:,-1,:]
        memory = memory[:,:-1,:]
        if query is None:
            query = last
        else:
            query = K.concatenate([query, last], axis=-1)

        if self.method is None:
            if len(input_shape) > 3:
                output_shape = K.int_shape(last)
                return K.reshape(last, (-1, input_shape[1], output_shape[-1]))
            else:
                return last
        elif self.method == 'cba':
            hidden = K.dot(memory, self.Wh) + K.expand_dims(K.dot(query, self.Wq), 1)
            hidden = K.tanh(hidden)
            s = K.squeeze(K.dot(hidden, self.v), -1)
        elif self.method == 'ga':
            s = K.sum(K.expand_dims(K.dot(query, self.Wq), 1) * memory, axis=-1)
        else:
            s = K.squeeze(K.dot(memory, self.v), -1)

        s = K.softmax(s)
        if mask is not None:
            mask = mask[:,:-1]
            s *= K.cast(mask, dtype='float32')
            sum_by_time = K.sum(s, axis=-1, keepdims=True)
            s = s / (sum_by_time + K.epsilon())
        #return [K.concatenate([K.sum(memory * K.expand_dims(s), axis=1), last], axis=-1), s]
        result = K.concatenate([K.sum(memory * K.expand_dims(s), axis=1), last], axis=-1)
        if len(input_shape)>3:
            output_shape = K.int_shape(result)
            return K.reshape(result, (-1, input_shape[1], output_shape[-1]))
        else:
            return result

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            memory = inputs[0]
        else:
            memory = inputs
        if len(K.int_shape(memory)) > 3 and mask is not None:
            return K.all(mask, axis=-1)
        else:
            return None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            att_size = input_shape[0][-1]
            seq_len = input_shape[0][1]
            batch = input_shape[0][0]
        else:
            att_size = input_shape[-1]
            seq_len = input_shape[1]
            batch = input_shape[0]
        #shape2 = (batch, seq_len, 1)
        if len(input_shape)>3:
            if self.method is not None:
                shape1 = (batch, seq_len, att_size*2)
            else:
                shape1 = (batch, seq_len, att_size)
            #return [shape1, shape2]
            return shape1
        else:
            if self.method is not None:
                shape1 = (batch, att_size*2)
            else:
                shape1 = (batch, att_size)
            #return [shape1, shape2]
            return shape1
def train_and_test_simple(data,targets,history_path,model_path,units_dim):
    train_in_c,test_in_c,train_out,test_out,train_in_d,test_in_d=data
    # print(train_in_d.shape,test_in_d.shape)
    start=(train_dim-look_back_span)*(train_days-test_days)
    validate_in_c=train_in_c[start:]
    validate_in_d=train_in_d[start:]
    validate_out=train_out[start:]
    
    train_in_new_c=train_in_c[:start]
    train_in_new_d=train_in_d[:start]
    train_out_new=train_out[:start]
    
    train_in_new_d_split=[train_in_new_d[:,j,:,:] for j in range(look_back_days)]
    validate_in_d_split=[validate_in_d[:,j,:,:] for j in range(look_back_days)]
    test_in_d_split=[test_in_d[:,j,:,:] for j in range(look_back_days)]    
    
    model_in=[train_in_new_c]+train_in_new_d_split
    model_out=[train_out_new]
    test_model_in=[test_in_c]+test_in_d_split
    test_model_out=[test_out]
    validate_in=[validate_in_c]+validate_in_d_split
    validate_out=[validate_out]
    fix_seeds(0)
    model=create_model(look_back_span,shift,len(targets),look_back_days,units_dim)
    model.summary()
    model.compile(loss='mse', optimizer = 'adam') 
    model.fit(x=model_in,y=model_out,batch_size=10,epochs=100,validation_data=(validate_in,validate_out),callbacks=[early_stopping],verbose=0) 
    # with open(history_path, 'wb') as file_pi:
    #     pickle.dump(train_history.history,file_pi)
    # model.save(model_path)
    model.save_weights(model_path)
    mape_tmp,gt,pred=evaluate_stdn(model,targets,test_model_in,test_model_out)
    return mape_tmp

def main():
    method,threshold,clusters,name=get_arguments()
    units_dim=8
    mark='stdn_{}_{}_{}'.format(method,threshold,units_dim)
    # print(test_days,train_days,train_dim,test_dim)
    # exit(0)
    mape=[]
    for index,c in enumerate(clusters):
        print('---Cluster {}-----'.format(index))
        print(c)
        # train_in_c,test_in_c,train_out_c,test_out_c,train_in_d,test_in_d=get_input(c)
        data=get_input(c)
        history_path='/root/yiwei_2021_05/data/history/{}_{}.txt'.format(mark,index)
        model_path='models/{}_{}_weights.h5'.format(mark,index)
        result_stdn=train_and_test_simple(data,c,history_path,model_path,units_dim)
        mape.append(result_stdn)
    mape_result=mape[0]
    for i in range(1,len(mape)):
        mape_result=numpy.concatenate((mape_result,mape[i]),axis=1)
    with open('result/{}.txt'.format(mark),'w') as f:
        f.write('{}:{}%\n'.format(mape_result.shape,numpy.around(numpy.mean(mape_result)*100,2)))
if __name__ == "__main__":
    main()