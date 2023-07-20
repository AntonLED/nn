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
                            initial_value=init(shape=(input_shape[-1], self.units), dtype=tf.float32), 
                            trainable=True)   
        M_hat = tf.Variable(name="M_hat",
                            initial_value=init(shape=(input_shape[-1], self.units), dtype=tf.float32),
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
                            initial_value=init(shape=(input_shape[-1], self.units), dtype=tf.float32), 
                            trainable=True)   
        M_hat = tf.Variable(name="M_hat",
                            initial_value=init(shape=(input_shape[-1], self.units), dtype=tf.float32),
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
                            initial_value=init(shape=(input_shape[-1], self.units), dtype=tf.float32), 
                            trainable=True)   
        self.nac_smpl = Nac_smpl_layer(self.units)
        self.nac_cmpx = Nac_cmpx_layer(self.units)
        self.g = G
    
    def call(self, inputs):
        gg = tf.nn.sigmoid(tf.matmul(inputs, self.g))
        self.a_smpl = self.nac_smpl(inputs)
        self.a_cmpx = self.nac_cmpx(inputs)
        return gg * self.a_smpl + (1 - gg) * self.a_cmpx

        
# data load
x_train = np.loadtxt("./datasets/data/in", dtype=np.float64)
y_train = np.loadtxt("./datasets/data/out", dtype=np.float64)

# data normalization
x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
y_train = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))

# model building
model = keras.Sequential([
    keras.layers.Input(1),
    # Nalu_layer(units=32),
    keras.layers.Dense(13, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.mean_squared_error)
model.fit(x_train, y_train, epochs=80, verbose=2, shuffle=False)

print(model.predict(x_train))
print(y_train)



