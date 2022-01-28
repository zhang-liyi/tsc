'''
Defines normalizing flows
'''



import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



# Real NVP
def Coupling_t(input_shape):
    input = tfk.Input(shape=input_shape)
    output_dim = input_shape * 5

    t_layer_1 = tfa.layers.WeightNormalization(tfkl.Dense(output_dim, activation="elu"))(input)
    t_layer_1 = tfkl.BatchNormalization()(t_layer_1)
    t_layer_2 = tfa.layers.WeightNormalization(tfkl.Dense(output_dim, activation="elu"))(t_layer_1)
    t_layer_2 = tfkl.BatchNormalization()(t_layer_2)
    t_layer_3 = tfa.layers.WeightNormalization(tfkl.Dense(input_shape, activation="linear"))(t_layer_2)

    return tf.keras.models.Model(inputs=input, outputs=t_layer_3)

def Coupling_s(input_shape):
    input = tfk.Input(shape=input_shape)
    output_dim = input_shape * 5

    s_layer_1 = tfa.layers.WeightNormalization(tfkl.Dense(output_dim, activation="elu"))(input)
    s_layer_1 = tfkl.BatchNormalization()(s_layer_1)
    s_layer_2 = tfa.layers.WeightNormalization(tfkl.Dense(output_dim, activation="elu"))(s_layer_1)
    s_layer_2 = tfkl.BatchNormalization()(s_layer_2)
    s_layer_3 = tfa.layers.WeightNormalization(tfkl.Dense(input_shape, activation="tanh"))(s_layer_2)

    return tf.keras.models.Model(inputs=input, outputs=s_layer_3)
    
class RealNVP(tf.keras.models.Model):
    def __init__(self, num_coupling_layers, num_dims):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers
        self.num_dims = num_dims
        
        if self.num_coupling_layers > 0:
            self.masks = np.zeros((self.num_dims, self.num_dims), dtype='float32')
            self.masks[1::2, ::2] = 1
            self.masks[::2, 1::2] = 1
            self.masks = np.tile(self.masks, (num_coupling_layers // 2, 1))

            self.layers_list_s = [Coupling_s(self.num_dims) for i in range(num_coupling_layers)]
            self.layers_list_t = [Coupling_t(self.num_dims) for i in range(num_coupling_layers)]

    def call(self, z):
        x = z
        logdet = 0.
        if self.num_coupling_layers > 0:
            for i in range(self.num_coupling_layers):
                x_masked = x * self.masks[i]
                s = self.layers_list_s[i](x_masked)
                t = self.layers_list_t[i](x_masked)
                s = s * (1 - self.masks[i]) 
                t = t * (1 - self.masks[i])
                x = x_masked + (1 - self.masks[i]) * (x * tf.exp(s) + t)
                logdet = logdet + tf.reduce_sum(s, axis=1)

        return x, logdet

    def inverse(self, x):
        z = x
        logdetinv = 0.
        if self.num_coupling_layers > 0:
            for i in reversed(range(self.num_coupling_layers)):
                z_masked = self.masks[i] * z
                s = self.layers_list_s[i](z_masked)
                t = self.layers_list_t[i](z_masked)
                s = s * (1 - self.masks[i])
                t = t * (1 - self.masks[i])
                z = z_masked + (1 - self.masks[i]) * (z - t) * tf.exp(-s)
                logdetinv = logdetinv - tf.reduce_sum(s, axis=1)

        return z, logdetinv

    

class Affine(tf.keras.models.Model):
    def __init__(self, num_coupling_layers, num_dims):
        super(Affine, self).__init__()

        self.num_dims = num_dims
        self.mu = tf.Variable(tf.zeros(self.num_dims), dtype=tf.float32)

    def call(self, z):
        x = z
        logdet = 0.
       
        return x, logdet

    def inverse(self, x):
        z = x
        logdetinv = 0.

        return z, logdetinv