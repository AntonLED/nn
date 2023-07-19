import tensorflow as tf
from tensorflow import keras
import numpy as np 


class Nac_cmpx_layer(keras.layers.Layer):                
    def __init__(self, units=32):                  
        super(Nac_cmpx_layer, self).__init__()            
        self.units = units   
                             
    def build(self, input_shape):        
        init = tf.random_normal_initializer() 
        W_hat = tf.Variable(name="W_hat",
                            initial_value=init(shape=(input_shape[-1], self.units), dtype="float32"), 
                            trainable=True)   
        M_hat = tf.Variable(name="M_hat",
                            initial_value=init(shape=(input_shape[-1], self.units), dtype="float32"),
                            trainable=True)         
        self.w = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    def call(self, inputs):
        return tf.exp(tf.matmul(tf.math.log(tf.abs(inputs) + 1e-7), self.w))
    

class Nac_smpl_layer(keras.layers.Layer):
    def __init__(self, units=32):
        super(Nac_smpl_layer, self).__init__()
        self.units = units 

    def build(self, input_shape):
        init = tf.random_normal_initializer() 
        W_hat = tf.Variable(name="W_hat",
                            initial_value=init(shape=(input_shape[-1], self.units), dtype="float32"), 
                            trainable=True)   
        M_hat = tf.Variable(name="M_hat",
                            initial_value=init(shape=(input_shape[-1], self.units), dtype="float32"),
                            trainable=True)         
        self.w = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class Nalu_layer(keras.layers.Layer):
    def __init__(self, units=32):
        super(Nalu_layer, self).__init__()
        self.units = units

    def build(self, input_shape):
        init = tf.random_normal_initializer() 
        G = tf.Variable(name="Gate_weights",
                            initial_value=init(shape=(input_shape[-1], self.units), dtype='float32'), 
                            trainable=True)   
        self.nac_smpl = Nac_smpl_layer(self.units)
        self.nac_cmpx = Nac_cmpx_layer(self.units)
        self.g = G
    
    def call(self, inputs):
        gg = tf.nn.sigmoid(tf.matmul(inputs, self.g))
        self.a_smpl = self.nac_smpl(inputs)
        self.a_cmpx = self.nac_cmpx(inputs)
        return gg * self.a_smpl + (1 - gg) * self.a_cmpx

        
x_train = np.arange(start=1, stop=100, step=1, dtype=np.int64)
y_train = x_train + 1

model = keras.Sequential([
    keras.layers.Input(1),
    Nac_smpl_layer(units=13),
    # Nac_cmpx_layer(units=10),
    # Nalu_layer(units=10),
    keras.layers.Dense(1)
])

# inputs = Nalu(units=1)
# # x = Nalu(units=1)(inputs)
# outputs = Nalu(units=32)(inputs)
# # outputs = keras.layers.Dense(1)(x)

# model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5000, verbose=2)

print(model.predict([9]))
print(model.predict([200]))
print(model.predict([200000]))
print(model.predict([200000]))
print(model.predict(np.array([3508733261694114463368191301286], dtype=np.float32)))
print("{:.0f}".format(str(model.predict(np.array([3508733261694114463368191301286], dtype=np.float32)))))



