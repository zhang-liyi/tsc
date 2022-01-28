import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def preprocess_mnist(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def load_image_data(dataset, batch_size=32):
    print('data processing...')

    if dataset == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = preprocess_mnist(train_images)
        test_images = preprocess_mnist(test_images)
        train_size = 60000
        test_size = 10000

    else:
        if dataset == 'mnist_dyn':
            train_size = 60000
            test_size = 10000

            ds_train, ds_test = tfds.load('binarized_mnist', split=['train', 'test'], shuffle_files=False)
            ds_train_np = tfds.as_numpy(ds_train)
            train_images = np.zeros((train_size, 28, 28, 1), dtype='float32')
            ds_test_np = tfds.as_numpy(ds_test)
            test_images = np.zeros((test_size, 28, 28, 1), dtype='float32')

        elif dataset == 'cifar10':

            train_size = 50000
            test_size = 10000

            ds_train, ds_test = tfds.load('cifar10', split=['train', 'test'], shuffle_files=False)
            ds_train_np = tfds.as_numpy(ds_train)
            train_images = np.zeros((train_size, 32, 32, 3), dtype='float32')
            ds_test_np = tfds.as_numpy(ds_test)
            test_images = np.zeros((test_size, 32, 32, 3), dtype='float32')

        i = 0
        for example in ds_train_np:
            train_images[i,:,:,:] = example['image']
            i += 1
        i = 0
        for example in ds_test_np:
            test_images[i,:,:,:] = example['image']
            i += 1

        if dataset == 'cifar10':
            train_images = (train_images - 255/2)/(255/2)
            test_images = (test_images - 255/2)/(255/2)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size, reshuffle_each_iteration=False, seed=0).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size, reshuffle_each_iteration=False, seed=0).batch(batch_size))

    return train_dataset, test_dataset, train_size, test_size

def generate_images_from_images(model, test_sample, colored=False, path=None):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    z, _ = model.flow_model(z)
    predictions = model.sample(z, apply_sigmoid=not colored)
    fig = plt.figure(figsize=(4, 4))
    if not colored:
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')
    else:
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i,:,:,:]+1)/2)
            plt.axis('off')
    if path is not None:
        plt.savefig(path)
        plt.close()
    
def generate_images_from_random(model, latent_dim, rand_vec=None, colored=False, path=None):
    if rand_vec is None:
        rand_vec = tfd.Sample(tfd.Normal(0., 1.), latent_dim).sample(16)
    predictions = model.sample(rand_vec, apply_sigmoid=not colored)
    fig = plt.figure(figsize=(4, 4))
    if not colored:
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')
    else:
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i,:,:,:]+1)/2)
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

def combine_nn_params(model, path, S=100):
    params = []
    model.load_weights(path + 'models/model0/model')
    num_params = len(model.trainable_variables)
    for param in model.trainable_variables:
        params.append(param)
    for s in range(1, S):
        model.load_weights(path + 'models/model{}/model'.format(s))
        for j in range(num_params):
            params[j] = params[j] + model.trainable_variables[j]
            if s == (S-1):
                params[j] = params[j] / S
                model.trainable_variables[j].assign(params[j])
    return model



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
























