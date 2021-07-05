from keras.layers import Dense, Activation, Input, Reshape,  Concatenate,GRU
from keras.models import Model
def create_stdn(time_steps_c,time_steps_d,dim,look_back_days,hidden_units):#dim in current day and past days are same here
    units_dim=hidden_units
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
