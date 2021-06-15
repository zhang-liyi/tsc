import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import datetime
import os

import util

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


class VAE(tf.keras.Model):
    """variational autoencoder."""

    def __init__(self, latent_dim, batch_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def train(self, train_dataset, test_dataset, epochs=10, lr=1e-4,
        test_sample=None, random_vector_for_generation=None, generation=False):
        self.write_results_helper('results/vae/')
        rp = open(self.file_path + "run_parameters.txt", "w")
        rp.write('Latent dimension: ' + str(self.latent_dim) + '\n')
        rp.write('Number of epochs: ' + str(epochs) + '\n')
        rp.write('Learning rate: ' + str(lr) + '\n')
        rp.write('Batch size: ' + str(self.batch_size) + '\n')
        rp.close()

        optimizer = tf.keras.optimizers.Adam(lr)
        for epoch in range(1, epochs + 1):

            start_time = datetime.datetime.now()
            for train_x in train_dataset:
                elbo = -self.train_step(train_x, optimizer)
            end_time = datetime.datetime.now()

            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(self.compute_loss(test_x))
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))

            if generation:
                util.generate_images_from_images(self, test_sample)
                util.generate_images_from_random(self, random_vector_for_generation)
            
    def write_results_helper(self, folder):
        # Save directory
        tm = str(datetime.datetime.now())
        tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
        path = folder + tm_str + '/' 
        self.file_path = path
        if not os.path.exists(path): 
            os.makedirs(path)
            os.makedirs(path+'generated-images/')
            os.makedirs(path+'encoder/')
            os.makedirs(path+'decoder/')
            for i in range(self.num_flow):
                os.makedirs(path+'arnet'+str(i)+'/')



class VAE_Flow(VAE):
    """variational autoencoder with a flow-based posterior."""

    def __init__(self, latent_dim, num_flow=5, batch_size=32):
        super().__init__(latent_dim, batch_size=batch_size)
        self.num_flow = num_flow
        self.arnets = []
        self.flows = []
        for _ in range(num_flow):
            arnet = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tfb.AutoregressiveNetwork(
                        params=2,
                        hidden_units=[latent_dim*5,latent_dim*5],
                        event_shape=self.latent_dim,
                        activation='elu',
                        kernel_initializer=tfk.initializers.GlorotNormal()),
                ]
            )
            self.arnets.append(arnet)
            self.flows.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(arnet)))
        self.f_tot = tfb.Chain(self.flows)

    def flow(self, x):
        return self.f_tot.forward(x)
    
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        zt = self.flow(z)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(zt, 0., 0.)
        q = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.exp(logvar/2)),
            bijector=self.f_tot)
        logqz_x = q.log_prob(zt)
        
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train(self, train_dataset, test_dataset, epochs=10, lr=1e-4,
        test_sample=None, random_vector_for_generation=None, generation=False):
        self.write_results_helper('results/vae_flow/')
        rp = open(self.file_path + "run_parameters.txt", "w")
        rp.write('Latent dimension: ' + str(self.latent_dim) + '\n')
        rp.write('Number of epochs: ' + str(epochs) + '\n')
        rp.write('Learning rate: ' + str(lr) + '\n')
        rp.write('Batch size: ' + str(self.batch_size) + '\n')
        rp.close()
        
        optimizer = tf.keras.optimizers.Adam(lr)
        for epoch in range(1, epochs + 1):
            start_time = datetime.datetime.now()
            for train_x in train_dataset:
                elbo = -self.train_step(train_x, optimizer)
            end_time = datetime.datetime.now()

            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(self.compute_loss(test_x))
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))

            # Save generated images
            if generation:
                util.generate_images_from_images(self, test_sample, 
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-images.png')
                util.generate_images_from_random(self, random_vector_for_generation, 
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-prior.png')
            # Save model
            if epoch == epochs:
                self.encoder.save_weights(self.file_path + 'encoder/encoder')
                self.decoder.save_weights(self.file_path + 'decoder/decoder')
                for i in range(self.num_flow):
                    self.arnets[i].save_weights(self.file_path +'arnet'+str(i)+'/arnet'+str(i))



class VAE_HSC(VAE_Flow):
    """variational autoencoder using Hamiltonian score climbing assisted by flows."""
    
    def __init__(self, latent_dim, num_flow=5, num_samp=1, hmc_e=0.25, hmc_L=4, 
                 batch_size=32, train_size=60000):
        super().__init__(latent_dim, num_flow=num_flow, batch_size=batch_size)
        self.num_samp = num_samp
        self.hmc_e = hmc_e
        self.hmc_L = hmc_L
        self.train_size = train_size
        self.pz = tfd.Sample(
            tfd.Normal(0., 1.), sample_shape=(latent_dim,))
        
        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.log_hmc_target,
            step_size=np.float32(self.hmc_e),
            num_leapfrog_steps=self.hmc_L,
            state_gradients_are_stopped=True)
        self.hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=self.hmc_kernel, 
            num_adaptation_steps=self.num_samp, 
            target_accept_prob=0.75,
            adaptation_rate=0.15)
        self.hmc_points = self.pz.sample(train_size).numpy()
        self.is_accepted = 0
        self.is_accepted_list = []
        self.first_epoch = False
        self.adapted_step_sizes = []
    
    def log_hmc_target(self, z0):
        # Unnormalized density of p(z0 | x)
        zt = self.f_tot.forward(z0)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=self.x_batch)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.pz.log_prob(zt)
        logdet = self.f_tot.forward_log_det_jacobian(z0, 1)

        return logpx_z + logpz + logdet
    
    def get_current_state(self, idx):
        if not self.first_epoch:
            if idx + self.batch_size >= self.train_size:
                return self.f_tot.inverse(self.hmc_points[idx:, :])
            else:
                return self.f_tot.inverse(self.hmc_points[idx:(idx+self.batch_size), :])
        else:
            if idx + self.batch_size >= self.train_size:
                return self.hmc_points[idx:, :]
            else:
                return self.hmc_points[idx:(idx+self.batch_size), :]
        # if not self.first_epoch:
        #     return self.q_base.sample(1)
        # else:
        #     if idx == 0:
        #         return self.hmc_points[idx:(idx+self.batch_size), :]
        #     else:
        #         return self.q_base.sample(1)
        
    def modify_current_state(self, idx, zt):
        if idx + self.batch_size >= self.train_size:
            self.hmc_points[idx:, :] = zt.numpy()
        else:
            self.hmc_points[idx:(idx+self.batch_size), :] = zt.numpy()
    
    def compute_loss(self, x, idx):
        mean, logvar = self.encode(x)
        self.q_base = tfd.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.exp(logvar/2))
        self.q = tfd.TransformedDistribution(
            distribution=self.q_base,
            bijector=self.f_tot)
        
        self.x_batch = x
        self.current_state_batch = tf.squeeze(self.get_current_state(idx))
        out = tfp.mcmc.sample_chain(
            1, self.current_state_batch, 
            previous_kernel_results=None, kernel=self.hmc_kernel,
            num_burnin_steps=self.num_samp, num_steps_between_results=0, 
            trace_fn=(lambda current_state, kernel_results: kernel_results), 
            return_final_kernel_results=False, seed=None, name=None)
        kernel_results = out[1]
        self.adapted_step_sizes.append(kernel_results.inner_results.accepted_results.step_size.numpy())
        z0 = tf.gather(out[0], 0)
        zt = self.f_tot.forward(z0)
        z0 = tf.stop_gradient(z0)
        zt = tf.stop_gradient(zt)

        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logqz_x = self.q.log_prob(zt)

        self.is_accepted = np.mean(np.squeeze(kernel_results.inner_results.is_accepted.numpy()))
        self.is_accepted_list.append(self.is_accepted)
        self.modify_current_state(idx, zt)
            
        return -tf.reduce_mean(logpx_z + logqz_x)
    
    def train_step(self, x, optimizer, idx):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, idx)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    def train(self, train_dataset, test_dataset, epochs=10, lr=1e-4, stop_idx=60000,
        test_sample=None, random_vector_for_generation=None, generation=False):
        self.write_results_helper('results/vae_hsc/')
        rp = open(self.file_path + "run_parameters.txt", "w")
        rp.write('Latent dimension: ' + str(self.latent_dim) + '\n')
        rp.write('Number of epochs: ' + str(epochs) + '\n')
        rp.write('Learning rate: ' + str(lr) + '\n')
        rp.write('Batch size: ' + str(self.batch_size) + '\n')
        rp.write('HMC number of samples: ' + str(self.num_samp) + '\n')
        rp.write('HMC step size: ' + str(self.hmc_e) + '\n')
        rp.write('HMC number of leapfrog steps: ' + str(self.hmc_L) + '\n')
        rp.close()

        # Training
        optimizer = tf.keras.optimizers.Adam(lr)
        for epoch in range(1, epochs + 1):
            print('-- Epoch {} --'.format(epoch))
            start_time = datetime.datetime.now()
            idx = 0
            self.first_epoch = (epoch==1)
            for train_x in train_dataset:
                if idx >= stop_idx:
                    break
                loss = self.train_step(train_x, optimizer, idx)
                if idx % (self.batch_size*1) == 0:
                    print('Loss', round(loss.numpy(), 3), 
                          'acceptance rate', round(self.is_accepted, 3))
                idx += self.batch_size
            end_time = datetime.datetime.now()
            print('adapted step sizes of the first and last batches: \n', 
                self.adapted_step_sizes[0], self.adapted_step_sizes[-1])
            self.adapted_step_sizes = []

            # Save generated images
            if generation:
                util.generate_images_from_images(self, test_sample, 
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-images.png')
                util.generate_images_from_random(self, random_vector_for_generation, 
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-prior.png')
            # Save model
            if epoch % 5 == 0 or epoch == epochs:
                self.encoder.save_weights(self.file_path + 'encoder/encoder')
                self.decoder.save_weights(self.file_path + 'decoder/decoder')
                for i in range(self.num_flow):
                    self.arnets[i].save_weights(self.file_path +'arnet'+str(i)+'/arnet'+str(i))
                np.savetxt(self.file_path + 'hmc_acceptance.csv', np.squeeze(self.is_accepted_list))

        return






