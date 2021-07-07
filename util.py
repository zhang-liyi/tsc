import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def load_mnist(batch_size=32):
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

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

def generate_images_from_images(model, test_sample, path=None):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    z = model.f_tot.forward(z)
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

