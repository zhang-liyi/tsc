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

def load_iaf_model_and_dist(path, num_dims=2):
    base_distribution = tfd.Sample(
        tfd.Normal(0., 1.), sample_shape=[num_dims])
    made_qp = tfb.AutoregressiveNetwork(
        params=2,
        hidden_units=[20, 20],
        event_shape=(2,),
        activation='elu',
        kernel_initializer=tfk.initializers.GlorotNormal())
    x_in = tfkl.Input(shape=(2,), dtype=tf.float32) # eps
    x_ = made_qp(x_in)
    model = tfk.Model(x_in, x_)

    model.load_weights(path)
    bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(model))
    q = tfd.TransformedDistribution(
            base_distribution, 
            bij)

    return model, q

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def load_mnist(batch_size=32, return_labels=False):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_size = 60000
    test_size = 10000

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size, reshuffle_each_iteration=False, seed=0).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size, reshuffle_each_iteration=False, seed=0).batch(batch_size))

    if not return_labels:
        return train_dataset, test_dataset
    else:
        return train_dataset, train_labels, test_dataset, test_labels

def load_mnist_dyn(batch_size=32):
    train_images = pd.read_pickle('data/mnist_dyn_train.pickle')
    test_images = pd.read_pickle('data/mnist_dyn_test.pickle')
    
    train_size = 60000
    test_size = 10000

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size, reshuffle_each_iteration=False, seed=0).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size, reshuffle_each_iteration=False, seed=0).batch(batch_size))

    return train_dataset, test_dataset

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def E_log_normal_hessian(mean, sig):
    d = mean.shape[0]
    diag_mumu = -1 / (sig**2)
    diag_sigsig = -2 / (sig**2)
    diag = tf.concat([diag_mumu, diag_sigsig], axis=0)
    zeros = tf.zeros((2*d, 2*d))
    hessian = tf.linalg.set_diag(zeros, diag)
    return hessian

def log_normal_hessian(sample, mean, sig):
    n = sample.shape[0]
    if len(mean.shape) < 2:
        mean = tf.expand_dims(mean, axis=0)
    if len(sig.shape) < 2:
        sig = tf.expand_dims(sig, axis=0)
    diag_mumu = -1 / (sig**2)
    diag_sigsig = 1 / (sig**2) - 3 / (sig**4) * (sample-mean)**2
    diag_musig = 2 / (sig**3) * (mean-sample)
    zeros = tf.zeros((n, sample.shape[1], sample.shape[1]))
    mat_mumu = tf.linalg.set_diag(zeros, tf.tile(diag_mumu, (n, 1)))
    mat_musig = tf.linalg.set_diag(zeros, diag_musig)
    mat_sigsig = tf.linalg.set_diag(zeros, diag_sigsig)
    mat_upper = tf.concat([mat_mumu, mat_musig], axis=2)
    mat_lower = tf.concat([mat_musig, mat_sigsig], axis=2)
    hessian = tf.concat([mat_upper, mat_lower], axis=1)
    return hessian

def generate_images_from_images(model, test_sample, path=None):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    z, _ = model.flow_model(z)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    if path is not None:
        plt.savefig(path)
        plt.close()
    
def generate_images_from_random(model, rand_vec, path=None):
    predictions = model.sample(rand_vec)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    if path is not None:
        plt.savefig(path)
        plt.close()

def plot_hmc_points(points, old_points=None, path=None):
    if old_points is None:
        plt.plot(points[0,0],points[0,1], 'o', label='pt 1')
        plt.plot(points[1,0],points[1,1], 'o', label='pt 2')
        plt.plot(points[2,0],points[2,1], 'o', label='pt 3')
        plt.legend()
        plt.savefig(path)
        plt.close()
        return np.expand_dims(points,0)
    else:
        points = np.concatenate((old_points, np.expand_dims(points, 0)))
        plt.plot(points[:,0,0], points[:,0,1], label='pt 1')
        plt.plot(points[:,1,0], points[:,1,1], label='pt 2')
        plt.plot(points[:,2,0], points[:,2,1], label='pt 3')
        plt.legend()
        plt.savefig(path)
        plt.close()
        return points
    

class Funnel:

    def __init__(self, num_dims=2):
        self.num_dims = num_dims

    # Define funnel distribution target
    def funnel_forward(self, x):
        shift = tf.zeros_like(x)
        log_scale = tf.concat(
            [tf.zeros_like(x[:, :1]),
             tf.tile(x[:, :1], [1, self.num_dims - 1])], -1)
        return shift, log_scale

    def get_funnel_dist(self):
        mg = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.num_dims), scale_identity_multiplier=1.0)
        target = tfd.TransformedDistribution(
            mg, bijector=tfb.MaskedAutoregressiveFlow(self.funnel_forward))

        return target

    def visualize_funnel_dist(self, target, s=10000):
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
    # (it is analytical, but for convenience, can approximate it as well)
    def estimate_funnal_dist(self, target, s=1e7):
        target_samp = target.sample(s)
        std = tf.math.reduce_std(target_samp, axis=0).numpy()
        m = tf.reduce_mean(target_samp, axis=0).numpy()
        return [m, std]



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

    


























