import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import datetime
import os
from copy import deepcopy

from util import *
from dist import *
from flow import *
from network import * 

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class VAE(tf.keras.Model):
    """variational autoencoder, with the option of adding flow."""

    def __init__(self, latent_dim, num_flow=0, batch_size=32, K=1, dataset='mnist', architecture='cnn', likelihood_sigma=1.):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_flow = num_flow
        self.batch_size = batch_size
        self.K = K # K = 1 is original VAE; K > 1 is IWAE
        self.pz = tfd.Sample(
            tfd.Normal(0., 1.), sample_shape=(latent_dim,))
        self.likelihood_sigma = likelihood_sigma # Only used when likelihood is Gaussian
        self.dataset = dataset
        # Define architectures: self.encoder & self.decoder.
        # The final layers should not have activations. Sigmoid, if needed,
        # should be included in later compute_loss or other functions.
        if self.dataset == 'mnist' or self.dataset == 'mnist_dyn' or self.dataset == 'fashion_mnist':
            if architecture == 'cnn':
                self.encoder = encoder_cnn_small(self.latent_dim)
                self.decoder = decoder_cnn_small(self.latent_dim)
            else:
                self.encoder = encoder_dense(self.latent_dim)
                self.decoder = decoder_dense(self.latent_dim)
        elif self.dataset == 'cifar10':
            self.encoder = DCGANEncoder(self.latent_dim)
            self.decoder = DCGANDecoder(self.latent_dim)
        self.flow_model = RealNVP(
            self.num_flow, 
            self.latent_dim)

    def sample(self, eps=None, apply_sigmoid=False):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=apply_sigmoid)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        x_dec = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(x_dec)
            return probs
        return x_dec

    def log_joint(self, z):
        x_dec = self.decode(z)
        if self.dataset == 'mnist' or self.dataset == 'mnist_dyn':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_dec, 
                labels=self.x_batch)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif self.dataset == 'fashion_mnist' or self.dataset == 'cifar10':
            logpx_z = log_normal_pdf(self.x_batch, 
                x_dec, 
                tf.math.log(self.likelihood_sigma**2), 
                raxis=range(1, len(self.x_batch.shape)))
        logpz = self.pz.log_prob(z)

        return logpx_z + logpz

    def log_var_dist(self, z):
        mean, logvar = self.encode(self.x_batch)
        z0, logdetinv = self.flow_model.inverse(z)
        logqz_x = log_normal_pdf(z0, mean, logvar) + logdetinv

        return logqz_x

    def compute_loss(self, x):
        # Encode
        mean, logvar = self.encode(x)
        z0 = self.reparameterize(
            tf.tile(mean, (self.K, 1)), 
            tf.tile(logvar, (self.K, 1)))
        zt, logdet = self.flow_model(z0)
        # Decode
        x_dec = self.decode(zt)
        # Compute loss
        if self.dataset == 'mnist' or self.dataset == 'mnist_dyn':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_dec, 
                labels=tf.tile(x, (self.K, 1, 1, 1)))
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif self.dataset == 'fashion_mnist' or self.dataset == 'cifar10':
            logpx_z = log_normal_pdf(tf.tile(x, (self.K, 1, 1, 1)), 
                x_dec, 
                tf.math.log(self.likelihood_sigma**2), 
                raxis=range(1, len(x.shape)))
        logpz = log_normal_pdf(zt, 0., 0.)
        logqz_x = -logdet + log_normal_pdf(
            z0, 
            tf.tile(mean, (self.K, 1)), 
            tf.tile(logvar, (self.K, 1)))
        logpx_z = tf.reshape(logpx_z, (self.K, -1))
        logpz = tf.reshape(logpz, (self.K, -1))
        logqz_x = tf.reshape(logqz_x, (self.K, -1))

        # Return the IWAE objective (same as original VAE if K = 0)
        loss_across_batches = -tf.math.reduce_logsumexp(logpx_z + logpz - logqz_x, axis=0) + tf.math.log(tf.cast(self.K, tf.float32))
        return tf.reduce_mean(loss_across_batches)

    def train_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def train(self, train_dataset, test_dataset, epochs=10, lr=1e-4, load_path=None, load_epoch=1, 
        test_sample=None, random_vector_for_generation=None, generation=False):
        self.write_results_helper('results/vae/')
        rp = open(self.file_path + "run_parameters.txt", "w")
        rp.write('Latent dimension: ' + str(self.latent_dim) + '\n')
        rp.write('Number of flows: ' + str(self.num_flow) + '\n')
        rp.write('Number of epochs: ' + str(epochs) + '\n')
        rp.write('VI number of samples (K): ' + str(self.K) + '\n')
        rp.write('Learning rate: ' + str(lr) + '\n')
        rp.write('Batch size: ' + str(self.batch_size) + '\n')
        rp.close()

        if load_path is not None:
            self.encoder.load_weights(load_path + 'encoder/encoder')
            self.decoder.load_weights(load_path + 'decoder/decoder')
            if self.num_flow > 0:
                self.flow_model.load_weights(
                        load_path + 'flow/flow')
        
        optimizer = tf.keras.optimizers.Adam(lr)
        for epoch in range(load_epoch, epochs + 1):

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
            if self.dataset == 'cifar10':
                colored = True 
            else:
                colored = False
            if generation:
                generate_images_from_images(self, test_sample,
                    colored=colored,
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-images.png')
                generate_images_from_random(self, self.latent_dim,
                    random_vector_for_generation, 
                    colored=colored,
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-prior.png')
            # Save model
            if epoch % 5 == 0 or epoch == epochs:
                self.encoder.save_weights(self.file_path + 'encoder/encoder')
                self.decoder.save_weights(self.file_path + 'decoder/decoder')
                self.flow_model.save_weights(self.file_path +'flow/flow')
            
    def write_results_helper(self, folder):
        # Save directory
        tm = str(datetime.datetime.now())
        tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
        path = folder + tm_str + '/' 
        self.file_path = path
        if not os.path.exists(path): 
            os.makedirs(path)
            os.makedirs(path+'generated-images/')
            os.makedirs(path+'hmc-points/')
            os.makedirs(path+'encoder/')
            os.makedirs(path+'decoder/')
            os.makedirs(path+'flow/')
        self.tm_str = tm_str



class VAE_HSC(VAE):
    """
    Variational autoencoder using Hamiltonian score climbing.
    It includes:
    * MSC (space='original')
    * HSC (space='warped' or 'eps')
    * CIS-MSC (space='original' and cis > 0)
    """
    
    def __init__(self, latent_dim, num_flow=5, space='original', dataset='mnist',
                 architecture='cnn', likelihood_sigma=1., cis=0, num_samp=1, chains=1, hmc_e=0.25, hmc_L=4, hmc_L_cap=4,
                 q_factor=1., batch_size=32, train_size=60000, target_accept=0.67, 
                 hmc_e_differs=False, reinitialize_from_q=False, shear=True):
        super().__init__(latent_dim, num_flow=num_flow, batch_size=batch_size,
            K=1, dataset=dataset, architecture=architecture, likelihood_sigma=likelihood_sigma)
        self.space = space
        self.cis = cis
        self.num_samp = num_samp
        self.chains = chains
        self.hmc_e_differs = hmc_e_differs
        self.hmc_e = hmc_e
        self.hmc_L = hmc_L # self.hmc_L = min(max(int(1 / hmc_e), 1), 25)
        self.hmc_L_cap = hmc_L_cap
        self.target_accept = target_accept
        self.q_factor = q_factor
        self.q_factor_variable = self.q_factor < 0
        self.train_size = train_size
        self.shear = shear
        self.reinitialize_from_q = reinitialize_from_q
        self.pz = tfd.Sample(
            tfd.Normal(0., 1.), sample_shape=(latent_dim,))
        self.get_shearing_param(shear)
        
        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.log_hmc_target,
            step_size=np.float32(self.hmc_e),
            num_leapfrog_steps=self.hmc_L,
            step_size_update_fn=None,
            state_gradients_are_stopped=True)
        self.hmc_points = self.pz.sample(self.chains*self.train_size).numpy()
        self.is_accepted_all_epochs = []
        self.is_accepted = [1.]
        self.epoch = 999

    def sample(self, eps=None, apply_sigmoid=False):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=apply_sigmoid)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        A = self.get_shearing_matrix()
        z = tf.matmul(z, tf.transpose(A))
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def get_shearing_param(self, shear=False):
        l = tf.cast((self.latent_dim+1)*self.latent_dim/2, tf.int32)
        self.lower_tri = tfp.math.fill_triangular(
            tf.ones(l))
        self.shearing_param = tf.Variable(
            tf.zeros((self.latent_dim, self.latent_dim)) + 0.1, 
            dtype=tf.float32,
            name='shear')
        return

    def get_shearing_matrix(self):
        if self.shear:
            return tf.linalg.set_diag(
                tf.multiply(self.lower_tri, self.shearing_param), [1]*self.latent_dim)
        else:
            return tf.eye(self.latent_dim)

    def log_joint(self, z):
        x_dec = self.decode(z)
        if self.dataset == 'mnist' or self.dataset == 'mnist_dyn':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_dec, 
                labels=tf.tile(self.x_batch, (self.chains, 1, 1, 1)))
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif self.dataset == 'fashion_mnist' or self.dataset == 'cifar10':
            logpx_z = log_normal_pdf(tf.tile(self.x_batch, (self.chains, 1, 1, 1)), 
                x_dec, 
                tf.math.log(self.likelihood_sigma**2), 
                raxis=range(1, len(self.x_batch.shape)))
        logpz = self.pz.log_prob(z)

        return logpx_z + logpz

    def log_var_dist(self, z):
        mean, logvar = self.encode(self.x_batch)
        z0, logdetinv = self.flow_model.inverse(z)
        logqz_x = logdetinv + log_normal_pdf(
            z0, tf.tile(mean, (self.chains, 1)), tf.tile(logvar, (self.chains, 1)))
        return logqz_x

    def log_hmc_target(self, z0):
        # For warped space: unnormalized density of p(z0 | x)
        # For original space: unnormalized density of p(zt | x)
        logdetjac = 0.
        if self.space == 'warped' or self.space == 'eps':
            z0 = self.mean + tf.multiply(z0, tf.exp(self.logvar / 2))
            logdetjac = logdetjac + tf.reduce_sum(self.logvar / 2, axis=1)
            zt, flow_logdet = self.flow_model(z0)
        else:
            zt = z0 
            flow_logdet = 0.
        x_dec = self.decode(zt)
        if self.dataset == 'mnist' or self.dataset == 'mnist_dyn':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_dec, 
                labels=tf.tile(self.x_batch, (self.chains, 1, 1, 1)))
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif self.dataset == 'fashion_mnist' or self.dataset == 'cifar10':
            logpx_z = log_normal_pdf(tf.tile(self.x_batch, (self.chains, 1, 1, 1)), 
                x_dec, 
                tf.math.log(self.likelihood_sigma**2), 
                raxis=range(1, len(self.x_batch.shape)))
        logpz = self.pz.log_prob(zt)
            
        logdetjac = logdetjac + flow_logdet

        return logpx_z + logpz + logdetjac

    def reset_hmc_kernel(self):
        smallest_accept = np.min(self.is_accepted)
        average_accept = np.mean(self.is_accepted)
        if smallest_accept < 0.25 or average_accept < self.target_accept:
            self.hmc_e = self.hmc_e * 0.97
        else:
            self.hmc_e = min(self.hmc_e * 1.03, 1.)
        self.hmc_L = min(int(1 / self.hmc_e)+1, self.hmc_L_cap)

        if self.hmc_e_differs:
            self.hmc_e_M = tf.exp(self.logvar / 2) * self.hmc_e
            self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.log_hmc_target,
                step_size=np.float32(self.hmc_e_M),
                num_leapfrog_steps=self.hmc_L,
                state_gradients_are_stopped=True)
        else:
            self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.log_hmc_target,
                step_size=np.float32(self.hmc_e),
                num_leapfrog_steps=self.hmc_L,
                state_gradients_are_stopped=True)

    def get_current_state(self, idx):  
        # Return the current state for HMC to sample
        if self.space == 'warped' or self.space == 'eps':
            if self.reinitialize_from_q:         
                return tf.stop_gradient(
                    (self.z0_from_q - self.mean) / tf.exp(self.logvar / 2))
            else:
                if self.epoch == 1:
                    return tf.stop_gradient(
                        (self.z0_from_q - self.mean) / tf.exp(self.logvar / 2))
                else:
                    zt_from_current_state = self.hmc_points[idx:(idx+self.batch_size*self.chains), :]
                    z0, _ = self.flow_model.inverse(zt_from_current_state)
                    return tf.stop_gradient(
                        (z0 - self.mean) / tf.exp(self.logvar / 2))
        else:
            if self.reinitialize_from_q:     
                zt, _ = self.flow_model(self.z0_from_q)    
                return zt
            else:
                if self.epoch == 1:
                    zt, _ = self.flow_model(self.z0_from_q)
                    return zt
                else:
                    zt = self.hmc_points[idx:(idx+self.batch_size*self.chains), :]
                    return zt
            
    def modify_current_state(self, idx, zt):
        # Record zt to use as 'current state' for HMC sampling in the next epoch
        if idx / self.chains + self.batch_size >= self.train_size:
            self.hmc_points[idx:, :] = zt.numpy()
        else:
            self.hmc_points[idx:(idx+self.batch_size*self.chains), :] = zt.numpy()

    def get_loss_q_from_q_sample(self, x):
        # Only used in NeutraHMC main training phase
        z0_from_q = self.reparameterize(
            self.mean, 
            self.logvar)
        zt_from_q, logdet_from_q = self.flow_model(z0_from_q)
        x_dec_from_q = self.decode(zt_from_q)
        if self.dataset == 'mnist' or self.dataset == 'mnist_dyn':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_dec_from_q, 
                labels=x)
            logpx_z_from_q = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif self.dataset == 'fashion_mnist' or self.dataset == 'cifar10':
            logpx_z_from_q = log_normal_pdf(x, 
                x_dec_from_q, 
                tf.math.log(self.likelihood_sigma**2), 
                raxis=range(1, len(x.shape)))
        logpz_from_q = log_normal_pdf(zt_from_q, 0., 0.)
        logqz_x_from_q = log_normal_pdf(
            z0_from_q, self.mean, self.logvar) - logdet_from_q

        loss_q = -tf.reduce_mean(logpx_z_from_q + logpz_from_q - logqz_x_from_q)

        return loss_q
    
    def compute_loss(self, x, idx):

        if self.cis == 0 or self.epoch == 1:
            mean, logvar = self.encode(x)
            self.mean = tf.tile(mean, (self.chains, 1))
            self.logvar = tf.tile(logvar, (self.chains, 1))
            self.z0_from_q = tf.stop_gradient(
                self.reparameterize(self.mean, self.logvar))

        self.x_batch = x
        self.current_state_batch = tf.squeeze(self.get_current_state(idx)) # batch_size by latent_dim

        # Get latents from HMC
        if self.cis == 0:
            self.reset_hmc_kernel()

            out = tfp.mcmc.sample_chain(
                self.num_samp, self.current_state_batch, 
                previous_kernel_results=None, kernel=self.hmc_kernel,
                num_burnin_steps=0, num_steps_between_results=0, 
                trace_fn=(lambda current_state, kernel_results: kernel_results.is_accepted), 
                return_final_kernel_results=False, seed=None, name=None)

            kernel_results = out[1]
            z0 = tf.gather(out[0], self.num_samp-1)
            self.is_accepted = np.mean(np.squeeze(kernel_results.numpy()), axis=0)
            self.is_accepted_all_epochs.append(np.mean(self.is_accepted))

            if self.space == 'warped' or self.space == 'eps':
                z0 = self.mean + tf.multiply(z0, tf.exp(self.logvar / 2))
                z0 = tf.stop_gradient(z0)
                zt, _ = self.flow_model(z0)
                zt = tf.stop_gradient(zt)  
            else:
                zt = tf.stop_gradient(z0)
        # Get latents from CIS-kernel MSC
        else:
            S = self.cis
            # Get mean, logvar from encoder
            self.mean, self.logvar = self.encode(x)
            # Get z for 2, 3, ..., S
            mean_Sm1 = tf.tile(self.mean, (S-1, 1))
            logvar_Sm1 = tf.tile(self.logvar, (S-1, 1))
            z0_Sm1 = tf.stop_gradient(
                self.reparameterize(mean_Sm1, logvar_Sm1)) 
            z0, logdetinv = self.flow_model.inverse(self.current_state_batch)
            z0_S = tf.concat([z0, z0_Sm1], axis=0)
            zt_Sm1, logdet_Sm1 = self.flow_model(z0_Sm1)
            # Concatenate z_Sm1 with z_k-1
            zt_S = tf.concat([self.current_state_batch, zt_Sm1], axis=0)
            logdet = tf.concat([-logdetinv, logdet_Sm1], axis=0)
            # Compute w as an S times batch_size-long vector
            x_dec = self.decode(zt_S)
            if self.dataset == 'mnist' or self.dataset == 'mnist_dyn':
                cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=x_dec, 
                    labels=tf.tile(x, (S, 1, 1, 1)))
                logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
            elif self.dataset == 'fashion_mnist' or self.dataset == 'cifar10':
                logpx_z = log_normal_pdf(tf.tile(x, (S, 1, 1, 1)), 
                    x_dec, 
                    tf.math.log(self.likelihood_sigma**2), 
                    raxis=range(1, len(x.shape)))
            logpz = log_normal_pdf(zt_S, 0., 0.)
            logqz_x = -logdet + log_normal_pdf(
                z0_S, 
                tf.tile(self.mean, (S, 1)), 
                tf.tile(self.logvar, (S, 1)))
            log_w = logpx_z + logpz - logqz_x
            # Reshape w into (batch_size, S)
            log_w = tf.transpose(tf.reshape(log_w, (S, -1)))
            # Sample from categorical (J is batch_size-long vector)
            J = tf.squeeze(tf.random.categorical(log_w, 1))
            # Set z[k], or z of this iteration
            zt_S = tf.reshape(zt_S, (S, -1, self.latent_dim))
            indices = tf.stack([J, tf.range(J.shape[0], dtype=tf.int64)], axis=0)
            indices = tf.transpose(indices)
            zt = tf.gather_nd(zt_S, indices)
            zt = tf.stop_gradient(zt)

            self.is_accepted_all_epochs = [1.] # trivial

        self.modify_current_state(idx, zt)

        z0, _ = self.flow_model.inverse(zt)
        _, logdet = self.flow_model(z0)
        x_dec = self.decode(zt)
        if self.dataset == 'mnist' or self.dataset == 'mnist_dyn':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_dec, 
                labels=tf.tile(x, (self.chains, 1, 1, 1)))
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif self.dataset == 'fashion_mnist' or self.dataset == 'cifar10':
            logpx_z = log_normal_pdf(
                tf.tile(x, (self.chains, 1, 1, 1)), 
                x_dec, 
                tf.math.log(self.likelihood_sigma**2), 
                raxis=range(1, len(x.shape)))
        logqz_x = log_normal_pdf(z0, self.mean, self.logvar) - logdet

        loss_p = -tf.reduce_mean(logpx_z)
        if not self.reinitialize_from_q:
            loss_q = -tf.reduce_mean(logqz_x)
        else:
            loss_q = self.get_loss_q_from_q_sample(x)

        print('loss_p', np.round(loss_p.numpy(), 3), 'loss_q', np.round(loss_q.numpy(), 3))
        
        return loss_p, loss_q

    def warm_up_compute_loss(self, x):
        mean, logvar = self.encode(x)
        z0 = self.reparameterize(mean, logvar)
        zt, logdet = self.flow_model(z0)
        x_dec = self.decode(zt)
        if self.dataset == 'mnist' or self.dataset == 'mnist_dyn':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_dec, 
                labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif self.dataset == 'fashion_mnist' or self.dataset == 'cifar10':
            logpx_z = log_normal_pdf(
                x, 
                x_dec, 
                tf.math.log(self.likelihood_sigma**2), 
                raxis=range(1, len(x.shape)))
        logpz = log_normal_pdf(zt, 0., 0.)
        logqz_x = log_normal_pdf(z0, mean, logvar) - logdet
        
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    def train_step(self, x, optimizer_p, optimizer_q, idx):

        with tf.GradientTape(persistent=True) as tape:
                
            loss_p, loss_q = self.compute_loss(x, idx)
            
        grads_p = tape.gradient(loss_p, self.decoder.trainable_variables)
        optimizer_p.apply_gradients(zip(grads_p, self.decoder.trainable_variables))

        if self.training_encoder:
            grads_q = tape.gradient(loss_q, self.encoder.trainable_variables + self.flow_model.trainable_variables + [self.shearing_param])
            optimizer_q.apply_gradients(zip(grads_q, self.encoder.trainable_variables + self.flow_model.trainable_variables + [self.shearing_param]))

        loss = loss_p + loss_q

        del tape
        
        return loss

    def warm_up_train_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.warm_up_compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def train(self, train_dataset, test_dataset, epochs=10, lr=1e-4, stop_idx=60000, 
        warm_up=False, train_encoder=True, q_not_train=0, load_path=None, load_epoch=1,
        test_sample=None, random_vector_for_generation=None, generation=False):

        self.write_results_helper('results/vae_hsc/')
        rp = open(self.file_path + "run_parameters.txt", "w")
        rp.write('Latent dimension: ' + str(self.latent_dim) + '\n')
        rp.write('Number of flows: ' + str(self.num_flow) + '\n')
        rp.write('Space: ' + self.space + '\n')
        rp.write('Number of epochs: ' + str(epochs) + '\n')
        rp.write('Learning rate: ' + str(lr) + '\n')
        rp.write('Batch size: ' + str(self.batch_size) + '\n')
        if self.cis != 0:
            rp.write('CIS samples: ' + str(self.cis) + '\n')
        rp.write('q factor: ' + str(self.q_factor) + '\n')
        rp.write('HMC number of samples: ' + str(self.num_samp) + '\n')
        rp.write('HMC number of chains: ' + str(self.chains) + '\n')
        rp.write('HMC step size: ' + str(self.hmc_e) + '\n')
        rp.write('HMC number of leapfrog steps: ' + str(self.hmc_L) + '\n')
        rp.write('Use shearing: ' + str(self.shear) + '\n')
        rp.write('Warm up with ELBO training: ' + str(warm_up) + '\n')
        rp.write('Reinitialize HMC from q (in every epoch): ' + str(self.reinitialize_from_q) + '\n')
        if not train_encoder:
            rp.write('Not training encoder during HSC')
        else:
            rp.write('q does not train for: ' + str(q_not_train))
        rp.close()

        # Training
        decay_change = (int(self.train_size/self.batch_size) + 1) * 400
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [decay_change], [lr, lr / 10])
        lr_schedule_q = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [decay_change], [lr * self.q_factor, lr * self.q_factor / 10])
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
        optimizer_q = tf.keras.optimizers.Adam(lr_schedule_q)
        hmc_zt = None
        losses = []
        self.decoder.save_weights(self.file_path + 'models/init_decoder/decoder')
        self.training_encoder = train_encoder
        reinitialize_from_q_setup = self.reinitialize_from_q

        # Load a pretrained encoder (and continue to train it). Meanwhile we want to train a decoder from scratch.
        if load_path is not None and load_epoch == 1:
            self.encoder.load_weights(load_path + 'warmed_up_encoder/encoder')
            if os.path.exists(load_path + 'warmed_up_decoder/'):
                self.decoder.load_weights(load_path + 'warmed_up_decoder/decoder')
            if self.num_flow > 0:
                self.flow_model.load_weights(load_path + 'warmed_up_flow/flow')
            if self.shear:
                self.shearing_param = tf.Variable(
                    np.genfromtxt(load_path + 'shearing_param.csv'),
                    dtype=tf.float32, 
                    name='shear')

        # Load a trained model and continue to train it. Epoch number adds cumulatively.
        elif load_path is not None:
            self.encoder.load_weights(load_path + 'models/epoch_' + str(load_epoch-1) + '_encoder/encoder')
            self.decoder.load_weights(load_path + 'models/epoch_' + str(load_epoch-1) + '_decoder/decoder')
            if self.num_flow > 0:
                self.flow_model.load_weights(
                        load_path + 'models/epoch_' + str(load_epoch-1) + '_flow/flow')
            if self.shear:
                self.shearing_param = tf.Variable(
                    np.genfromtxt(
                    load_path + 'models/epoch_' + str(load_epoch-1) + '_shearing_param/shearing_param.csv',
                    dtype='float32'),
                    dtype=tf.float32, 
                    name='shear')
            self.hmc_points = np.genfromtxt(load_path + 'hmc-points/hmc_points.csv', dtype='float32')

        # If we don't load, we do a warm-up and use the warmed-up encoder for NeutraHMC
        if warm_up and load_path is None:
            os.makedirs('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_encoder/')
            os.makedirs('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_flow/')
            if self.num_flow > 0:
                warm_up_epochs = 500
            else:
                warm_up_epochs = 20
            for epoch in range(1, warm_up_epochs + 1):
                print('-- Epoch (warm up) {} --'.format(epoch))
                for train_x in train_dataset:
                    loss = self.warm_up_train_step(train_x, optimizer)
                    print('ELBO', round(-loss.numpy(), 3))
                # Save generated images
                if generation:
                    generate_images_from_images(self, test_sample,
                        path=self.file_path + 'generated-images/pre-epoch-'+str(epoch)+'-from-images.png')
                    generate_images_from_random(self, self.latent_dim, 
                        random_vector_for_generation, 
                        path=self.file_path + 'generated-images/pre-epoch-'+str(epoch)+'-from-prior.png')
            # Save model
            self.encoder.save_weights(
                'pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_encoder/encoder')
            # self.decoder.save_weights(
            #     'pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_decoder/decoder')
            self.flow_model.save_weights(
                'pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_flow/flow')
            np.savetxt('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/shearing_param.csv', self.shearing_param.numpy())

            # End of warm up, we reload the initial decoder and be ready to start training
            self.decoder.load_weights(self.file_path + 'models/init_decoder/decoder')

        # Main section

        for epoch in range(load_epoch, epochs + 1):

            print('-- Epoch {} --'.format(epoch))
            idx = 0
            self.epoch = epoch

            # Decide whether the encoder should be trainable:
            if epoch <= q_not_train:
                self.encoder.trainable = False
                self.flow_model.trainable = False
                self.training_encoder = False
            else:
                self.encoder.trainable = train_encoder
                self.flow_model.trainable = train_encoder
                self.training_encoder = train_encoder

            # Training:
            for train_x in train_dataset:

                # We stop if we only want to train with stop_idx many data points:
                if idx >= stop_idx * self.chains:
                    break
                # Run a training step:
                start_time = datetime.datetime.now()
                loss = self.train_step(train_x, optimizer, optimizer_q, idx)
                # Update idx to record current_state for the next HMC step:
                idx += self.batch_size * self.chains
                losses.append(loss.numpy())
                end_time = datetime.datetime.now()
                
                if idx % (self.batch_size*1) == 0:
                    print('Loss', round(loss.numpy(), 3), 
                          'accept rate (avg)', round(self.is_accepted_all_epochs[-1], 3),
                          'step size', np.round(self.hmc_e, 3),
                          'time', end_time-start_time,
                          'idx', idx)
                    # print('Loss p', round(self.loss_p.numpy(), 3), 'Loss q', round(self.loss_q.numpy(), 3))

            # Save generated images
            if self.dataset == 'cifar10':
                colored = True 
            else:
                colored = False
            if generation:
                generate_images_from_images(self, test_sample,
                    colored=colored,
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-images.png')
                generate_images_from_random(self, self.latent_dim,
                    random_vector_for_generation, 
                    colored=colored,
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-prior.png')
            # Save model
            if epoch % 10 == 0 or epoch == epochs:
                os.makedirs(self.file_path+'models/epoch_' + str(epoch) + '_encoder/')
                os.makedirs(self.file_path+'models/epoch_' + str(epoch) + '_decoder/')
                os.makedirs(self.file_path+'models/epoch_' + str(epoch) + '_shearing_param/')
                self.encoder.save_weights(self.file_path+'models/epoch_' + str(epoch) + '_encoder/encoder')
                self.decoder.save_weights(self.file_path+'models/epoch_' + str(epoch) + '_decoder/decoder')
                np.savetxt(
                    self.file_path+'models/epoch_' + str(epoch) + '_shearing_param/shearing_param.csv', self.shearing_param)
                os.makedirs(self.file_path+'models/epoch_' + str(epoch) + '_flow/')
                self.flow_model.save_weights(
                    self.file_path+'models/epoch_' + str(epoch) + '_flow/flow')
                # np.savetxt(self.file_path + 'hmc_acceptance.csv', np.squeeze(self.is_accepted_list))
                np.savetxt(self.file_path + 'hmc-points/hmc_points.csv', np.squeeze(self.hmc_points))
                np.savetxt(self.file_path + 'losses.csv', np.array(losses))
            # Plot HMC points
            # if self.latent_dim == 2 and generation:
            #     if epoch == start_epoch:
            #         old_points = None
            #     old_points = plot_hmc_points(self.hmc_points[[0,1000,1999],:], old_points,
            #         path=self.file_path + 'hmc-points/epoch-'+str(epoch)+'.png')

        return






