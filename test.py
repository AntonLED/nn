import tensorflow as tf
from tensorflow import keras
import numpy as np 


class Nalu(keras.layers.Layer):                
    def __init__(self,units=32):                  
        super(Nalu, self).__init__()            
        self.units = units   
                             
    def build(self, input_shape):        
        init = tf.random_normal_initializer() 
        W_hat = tf.Variable(name="W_hat",
                            initial_value=init(shape=(input_shape[-1], self.units), dtype='float32'), 
                            trainable=True)   
        M_hat = tf.Variable(name="M_hat",
                            initial_value=init(shape=(input_shape[-1], self.units), dtype='float32'),
                            trainable=True)         
        self.w = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    def call(self,inputs):                        
        return tf.exp(tf.matmul(tf.math.log(tf.abs(inputs) + 1e-7), self.w))
    

x_train = np.arange(start=1, stop=1000, step=1, dtype=np.float32)
y_train = np.sqrt(x_train)

inputs = keras.Input(1)
x = Nalu(units=1)(inputs)
outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss='mean_squared_error')
model.fit(x_train, y_train, epochs=1000, verbose=2)

print(model.predict([9.0]))



