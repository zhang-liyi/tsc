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



class DCGANConv(tf.keras.layers.Layer):
    # Conv block to be used in DCGAN-style encoder
    # Each block consists of: Conv2D - BatchNorm - LeakyReLU
    def __init__(self, out_filter):
        super().__init__()
        self.conv2d = tf.keras.layers.Conv2D(out_filter, 
                                             (4, 4), 
                                             strides=(2, 2), 
                                             padding='same')
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.leakyrelu = tf.keras.layers.LeakyReLU(0.2)
        
    def call(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x
    
class DCGANConvT(tf.keras.layers.Layer):
    # ConvTranspose block to be used in DCGAN-style decoder
    # Each block consists of: Conv2DTranspose - BatchNorm - LeakyReLU
    def __init__(self, out_filter):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(out_filter, 
                                                               (4, 4), 
                                                               strides=(2, 2), 
                                                               padding='same')
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.leakyrelu = tf.keras.layers.LeakyReLU(0.2)
        
    def call(self, x):
        x = self.conv2dtranspose(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x

class DCGANEncoder(tfk.Model):
    # DCGAN-style encoder
    # It uses the previously defined DCGANConv blocks.
    def __init__(self, latent_dim, filter_size_init=64):
        super().__init__()
        self.nf = filter_size_init
        self.dcgan_conv1 = DCGANConv(self.nf)
        self.dcgan_conv2 = DCGANConv(self.nf*2)
        self.dcgan_conv3 = DCGANConv(self.nf*4)
        # self.dcgan_conv4 = DCGANConv(self.nf*8)
        self.conv_final = tf.keras.layers.Conv2D(latent_dim*2, 
                                                 (4, 4), 
                                                 strides=(1, 1), 
                                                 padding='valid')
        # self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):
        x = self.dcgan_conv1(x)
        x = self.dcgan_conv2(x)
        x = self.dcgan_conv3(x)
        # x = self.dcgan_conv4(x)
        x = self.conv_final(x)
        # x = self.batchnorm(x)
        x = self.flatten(x)
        return x

class DCGANDecoder(tfk.Model):
    
    def __init__(self, latent_dim, filter_size=64, output_channel=3):
        super().__init__()
        self.nf = filter_size # filter size of the second-to-last conv transpose layer
        self.nc = output_channel # 3 for CIFAR10
        
        self.reshape = tf.keras.layers.Reshape((1, 1, latent_dim))
        self.conv_t1 = tf.keras.layers.Conv2DTranspose(self.nf*4, 
                                                       (4, 4), 
                                                       strides=(1, 1), 
                                                       padding='valid')
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.leakyrelu = tf.keras.layers.LeakyReLU(0.2)
        self.dcgan_conv_t1 = DCGANConvT(self.nf*2)
        self.dcgan_conv_t2 = DCGANConvT(self.nf)
        self.conv_t2 = tf.keras.layers.Conv2DTranspose(self.nc, 
                                                       (4, 4), 
                                                       strides=(2, 2), 
                                                       padding='same',
                                                       activation='tanh')
        
    def call(self, x):
        x = self.reshape(x)
        x = self.conv_t1(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        x = self.dcgan_conv_t1(x)
        x = self.dcgan_conv_t2(x)
        x = self.conv_t2(x)
        return x



def encoder_cnn_small(latent_dim):
    encoder = tf.keras.Sequential(
                    [
                        tfkl.InputLayer(input_shape=(28, 28, 1)),
                        tfkl.Conv2D(
                            filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                        tfkl.Conv2D(
                            filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                        tfkl.Flatten(),
                        tfkl.Dense(latent_dim + latent_dim),
                    ]
                )
    return encoder

def decoder_cnn_small(latent_dim):
    decoder = tf.keras.Sequential(
                    [
                        tfkl.InputLayer(input_shape=(latent_dim,)),
                        tfkl.Dense(units=7*7*32, activation=tf.nn.relu),
                        tfkl.Reshape(target_shape=(7, 7, 32)),
                        tfkl.Conv2DTranspose(
                            filters=64, kernel_size=3, strides=2, padding='same',
                            activation='relu'),
                        tfkl.Conv2DTranspose(
                            filters=32, kernel_size=3, strides=2, padding='same',
                            activation='relu'),
                        tfkl.Conv2DTranspose(
                            filters=1, kernel_size=3, strides=1, padding='same'),
                    ]
                )
    return decoder 

def encoder_dense(latent_dim):
    encoder = tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                        tf.keras.layers.Reshape(target_shape=(28 * 28,)),
                        tf.keras.layers.Dense(1024, activation='relu'),
                        tf.keras.layers.Dense(1024, activation='relu'),
                        tf.keras.layers.Dense(latent_dim + latent_dim),
                    ]
                )
    return encoder

def decoder_dense(latent_dim):
    decoder = tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                        tf.keras.layers.Dense(1024, activation='relu'),
                        tf.keras.layers.Dense(1024, activation='relu'),
                        tf.keras.layers.Dense(28 * 28),
                        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
                    ]
                )
    return decoder




