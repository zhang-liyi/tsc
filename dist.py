'''
Defines distributions used in experiments:
* Funnel
* Banana
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



class Funnel:

    def __init__(self, num_dims=2):
        self.num_dims = num_dims

    def forward(self, x):
        shift = tf.zeros_like(x)
        log_scale = tf.concat(
            [tf.zeros_like(x[:, :1]),
             tf.tile(x[:, :1], [1, self.num_dims - 1])], -1)
        return shift, log_scale

    def get_dist(self):
        mg = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.num_dims), scale_identity_multiplier=1.0)
        target = tfd.TransformedDistribution(
            mg, bijector=tfb.MaskedAutoregressiveFlow(self.forward))

        return target

    def visualize_dist(self, target, s=10000):
        # Generate points from funnel distribution
        points = np.transpose(target.sample(s).numpy())
        Y = points[0]
        X = points[1]

        # Calculate the point density
        XY = np.vstack([X,Y])
        Z = gaussian_kde(XY)(XY)

        # Sort the points by density, so that the densest points are plotted last
        idx = Z.argsort()
        X, Y, Z = X[idx], Y[idx], Z[idx]

        plt.scatter(X, Y, c=Z, label=Z)
        plt.colorbar()
        plt.show()
        plt.close()

        return X, Y, Z

    # Estimate mean and std of the funnel distribution
    # (it is analytical, but for convenience, we approximate it as well)
    def estimate_dist(self, target, s=1e7):
        target_samp = target.sample(s)
        std = tf.math.reduce_std(target_samp, axis=0).numpy()
        m = tf.reduce_mean(target_samp, axis=0).numpy()
        return [m, std]



class Banana:

    def __init__(self, b=0.02):
        self.b = b

    def forward(self, z):
        z1, z2 = tf.split(z, [1,1], axis=-1)
        z1 = tf.squeeze(z1, axis=-1)
        z2 = tf.squeeze(z2, axis=-1)
        x1 = z1
        x2 = z2 + self.b*z1*z1 - 100*self.b
        return tf.stack([x1,x2], axis=-1)

    def inverse(self, x):
        x1, x2 = tf.split(x, [1,1], axis=-1)
        x1 = tf.squeeze(x1, axis=-1)
        x2 = tf.squeeze(x2, axis=-1)
        z1 = x1
        z2 = x2 - self.b*x1*x1 + 100*self.b
        return tf.stack([z1,z2], axis=-1)

    def get_dist(self):
        mg = tfd.MultivariateNormalDiag(loc=tf.zeros(2), 
            scale_diag=[10.,1.])

        bij = tfb.Inline(forward_fn=self.forward,
            inverse_fn=self.inverse,
            forward_log_det_jacobian_fn=lambda y: tf.zeros(1),
            inverse_log_det_jacobian_fn=lambda y: tf.zeros(1),
            forward_min_event_ndims=0)

        target = tfd.TransformedDistribution(
            mg, bijector=bij)

        return target

    def visualize_dist(self, target, s=10000):
        # Generate points from distribution
        points = np.transpose(target.sample(s).numpy())
        Y = points[0]
        X = points[1]

        # Calculate the point density
        XY = np.vstack([X,Y])
        Z = gaussian_kde(XY)(XY)

        # Sort the points by density, so that the densest points are plotted last
        idx = Z.argsort()
        X, Y, Z = X[idx], Y[idx], Z[idx]

        plt.scatter(X, Y, c=Z, label=Z)
        plt.colorbar()
        plt.show()
        plt.close()

        return X, Y, Z

    # Estimate mean and std of the distribution
    # (it is analytical, but for convenience, can approximate it as well)
    def estimate_dist(self, target, s=1e7):
        target_samp = target.sample(s)
        std = tf.math.reduce_std(target_samp, axis=0).numpy()
        m = tf.reduce_mean(target_samp, axis=0).numpy()
        return [m, std]









