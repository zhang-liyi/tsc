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
from copy import deepcopy

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


print('Running version ' + str(2))


class VAE(tf.keras.Model):
    """variational autoencoder, with the option of adding flow."""

    def __init__(self, latent_dim, num_flow=0, batch_size=32, num_samp=1, architecture='cnn'):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_flow = num_flow
        self.batch_size = batch_size
        self.num_samp = num_samp
        self.pz = tfd.Sample(
            tfd.Normal(0., 1.), sample_shape=(latent_dim,))
        if architecture == 'cnn':
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
                ]
            )
            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
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
        else:
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                    tf.keras.layers.Reshape(target_shape=(28 * 28,)),
                    tf.keras.layers.Dense(1024, activation='softplus'),
                    tf.keras.layers.Dense(1024, activation='softplus'),
                    tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
                ]
            )
            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                    tf.keras.layers.Dense(1024, activation='softplus'),
                    tf.keras.layers.Dense(1024, activation='softplus'),
                    tf.keras.layers.Dense(28 * 28),
                    tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
                ]
            )
        self.flow_model = util.RealNVP(
            self.num_flow, 
            self.latent_dim)

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

    def log_joint(self, z):
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=self.x_batch)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.pz.log_prob(z)

        return logpx_z + logpz

    def log_var_dist(self, z):
        mean, logvar = self.encode(self.x_batch)
        z0, logdetinv = self.flow_model.inverse(z)
        logqz_x = util.log_normal_pdf(z0, mean, logvar) + logdetinv

        return logqz_x

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z0 = self.reparameterize(
            tf.tile(mean, (self.num_samp, 1)), 
            tf.tile(logvar, (self.num_samp, 1)))
        zt, logdet = self.flow_model(z0)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, 
            labels=tf.tile(x, (self.num_samp, 1, 1, 1)))
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = util.log_normal_pdf(zt, 0., 0.)
        logqz_x = util.log_normal_pdf(
            z0, 
            tf.tile(mean, (self.num_samp, 1)), 
            tf.tile(logvar, (self.num_samp, 1))) \
            - logdet
        
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
        rp.write('Number of flows: ' + str(self.num_flow) + '\n')
        rp.write('Number of epochs: ' + str(epochs) + '\n')
        rp.write('VI number of samples: ' + str(self.num_samp) + '\n')
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
    """variational autoencoder using Hamiltonian score climbing."""
    
    def __init__(self, latent_dim, num_flow=5, space='original', architecture='cnn', 
                 num_samp=1, chains=1, hmc_e=0.25, hmc_L=4, q_factor=1.,
                 batch_size=32, train_size=60000, 
                 hmc_e_differs=False, reinitialize_from_q=False, shear=True):
        super().__init__(latent_dim, num_flow=num_flow, batch_size=batch_size)
        self.space = space
        self.num_samp = num_samp
        self.chains = chains
        self.hmc_e_differs = hmc_e_differs
        self.hmc_e = hmc_e
        self.hmc_L = hmc_L # self.hmc_L = min(max(int(1 / hmc_e), 1), 25)
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
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=tf.tile(self.x_batch, (self.chains, 1, 1, 1)))
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.pz.log_prob(z)

        return logpx_z + logpz

    def log_var_dist(self, z):
        mean, logvar = self.encode(self.x_batch)
        z0, logdetinv = self.flow_model.inverse(z)
        logqz_x = util.log_normal_pdf(
            z0, tf.tile(mean, (self.chains, 1)), tf.tile(logvar, (self.chains, 1))) \
            + logdetinv
        return logqz_x

    def log_hmc_target(self, z0):
        # Unnormalized density of p(z0 | x)
        logdetjac = 0.
        if self.space == 'warped' or self.space == 'eps':
            z0 = self.mean + tf.multiply(z0, tf.exp(self.logvar / 2))
            logdetjac = logdetjac + tf.reduce_sum(self.logvar / 2, axis=1)
        zt, flow_logdet = self.flow_model(z0)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=tf.tile(self.x_batch, (self.chains, 1, 1, 1)))
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.pz.log_prob(zt)
            
        logdetjac = logdetjac + flow_logdet

        return logpx_z + logpz + logdetjac

    def reset_hmc_kernel(self):
        smallest_accept = np.min(self.is_accepted)
        average_accept = np.mean(self.is_accepted)
        if smallest_accept < 0.25 or average_accept < 0.67:
            self.hmc_e = self.hmc_e * 0.97
        else:
            self.hmc_e = min(self.hmc_e * 1.03, 1.)
        self.hmc_L = min(int(1 / self.hmc_e)+1, 4)

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
                return self.z0_from_q
            else:
                if self.epoch == 1:
                    return self.z0_from_q
                else:
                    zt_from_current_state = self.hmc_points[idx:(idx+self.batch_size*self.chains), :]
                    z0, _ = self.flow_model.inverse(zt_from_current_state)
                    return tf.stop_gradient(z0)
            
    def modify_current_state(self, idx, zt):
        # Record zt to use as 'current state' for HMC sampling in the next epoch
        if idx / self.chains + self.batch_size >= self.train_size:
            self.hmc_points[idx:, :] = zt.numpy()
        else:
            self.hmc_points[idx:(idx+self.batch_size*self.chains), :] = zt.numpy()
    
    def compute_loss(self, x, idx):
        mean, logvar = self.encode(x)
        self.mean = tf.tile(mean, (self.chains, 1))
        self.logvar = tf.tile(logvar, (self.chains, 1))
        self.z0_from_q = tf.stop_gradient(
            self.reparameterize(self.mean, self.logvar))

        self.x_batch = x
        self.current_state_batch = tf.squeeze(self.get_current_state(idx))

        self.reset_hmc_kernel()
        out = tfp.mcmc.sample_chain(
            self.num_samp, self.current_state_batch, 
            previous_kernel_results=None, kernel=self.hmc_kernel,
            num_burnin_steps=0, num_steps_between_results=0, 
            trace_fn=(lambda current_state, kernel_results: kernel_results.is_accepted), 
            return_final_kernel_results=False, seed=None, name=None)
        kernel_results = out[1]
        z0 = tf.gather(out[0], self.num_samp-1)
        # all_z0 = tf.reshape(out[0], (-1, self.latent_dim))

        if self.space == 'warped' or self.space == 'eps':
            z0 = self.mean + tf.multiply(z0, tf.exp(self.logvar / 2))
            # all_z0 = tf.tile(self.mean, (self.num_samp, 1)) + \
            # tf.multiply(all_z0, tf.exp(tf.tile(self.logvar, (self.num_samp, 1)) / 2))
        z0 = tf.stop_gradient(z0)
        # all_z0 = tf.stop_gradient(all_z0)
        zt, _ = self.flow_model(z0)
        zt = tf.stop_gradient(zt)
        # all_zt, _ = self.flow_model(all_z0)
        # all_zt = tf.stop_gradient(all_zt)
        self.is_accepted = np.mean(np.squeeze(kernel_results.numpy()), axis=0)
        self.is_accepted_all_epochs.append(np.mean(self.is_accepted))
        self.modify_current_state(idx, zt)

        z0, _ = self.flow_model.inverse(zt)
        _, logdet = self.flow_model(z0)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=tf.tile(x, (self.chains, 1, 1, 1)))
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        # logqz_x = util.log_normal_pdf(all_z0, tf.tile(self.mean, (self.num_samp, 1)), tf.tile(self.logvar, (self.num_samp, 1))) - logdet
        logqz_x = util.log_normal_pdf(z0, tf.tile(self.mean, (1, 1)), tf.tile(self.logvar, (1, 1))) - logdet

        loss_p = -tf.reduce_mean(logpx_z)
        loss_q = -tf.reduce_mean(logqz_x)

        print('loss_p', np.round(loss_p.numpy(), 3), 'loss_q', np.round(loss_q.numpy(), 3))
        
        return loss_p, loss_q

    def warm_up_compute_loss(self, x):
        mean, logvar = self.encode(x)
        z0 = self.reparameterize(mean, logvar)
        zt, logdet = self.flow_model(z0)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = util.log_normal_pdf(zt, 0., 0.)
        logqz_x = util.log_normal_pdf(z0, mean, logvar) - logdet
        
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    def train_step(self, x, optimizer_p, optimizer_q, idx):
        with tf.GradientTape(persistent=True) as tape:
                
            loss_p, loss_q = self.compute_loss(x, idx)
            
        grads_p = tape.gradient(loss_p, self.decoder.trainable_variables)
        optimizer_p.apply_gradients(zip(grads_p, self.decoder.trainable_variables))

        if self.training_encoder:
            grads_q = tape.gradient(loss_q, self.encoder.trainable_variables + self.flow_model.trainable_variables)
            optimizer_q.apply_gradients(zip(grads_q, self.encoder.trainable_variables + self.flow_model.trainable_variables))

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
        optimizer = tf.keras.optimizers.Adam(lr)
        optimizer_q = tf.keras.optimizers.Adam(lr * self.q_factor)
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
                self.shearing_param = np.genfromtxt(load_path + 'shearing_param.csv')

        # Load a trained model and continue to train it. Epoch number adds cumulatively.
        elif load_path is not None:
            self.encoder.load_weights(load_path + 'models/epoch_' + str(load_epoch-1) + '_encoder/encoder')
            self.decoder.load_weights(load_path + 'models/epoch_' + str(load_epoch-1) + '_decoder/decoder')
            if self.num_flow > 0:
                self.flow_model.load_weights(
                        load_path + 'models/epoch_' + str(load_epoch-1) + '_flow/flow')
            if self.shear:
                self.shearing_param = np.genfromtxt(
                    load_path + 'models/epoch_' + str(load_epoch-1) + '_shearing_param/shearing_param.csv',
                    dtype='float32')
            self.hmc_points = np.genfromtxt(load_path + 'hmc-points/hmc_points.csv', dtype='float32')

        # If we don't load, we can do a warm-up and use the warmed-up encoder.
        if warm_up:
            os.makedirs('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_encoder/')
            os.makedirs('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_flow/')
            if self.num_flow > 0:
                warm_up_epochs = 100
            else:
                warm_up_epochs = 20
            for epoch in range(1, warm_up_epochs + 1):
                print('-- Epoch (warm up) {} --'.format(epoch))
                for train_x in train_dataset:
                    loss = self.warm_up_train_step(train_x, optimizer)
                    print('ELBO', round(-loss.numpy(), 3))
                # Save generated images
                if generation:
                    util.generate_images_from_images(self, test_sample,
                        path=self.file_path + 'generated-images/pre-epoch-'+str(epoch)+'-from-images.png')
                    util.generate_images_from_random(self, random_vector_for_generation, 
                        path=self.file_path + 'generated-images/pre-epoch-'+str(epoch)+'-from-prior.png')
            self.shearing_param = self.shearing_param.numpy()
            # Save model
            self.encoder.save_weights(
                'pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_encoder/encoder')
            # self.decoder.save_weights(
            #     'pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_decoder/decoder')
            self.flow_model.save_weights(
                'pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_flow/flow')
            np.savetxt('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/shearing_param.csv', self.shearing_param)

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

            # if epoch <= 20:
            #     self.reinitialize_from_q = True 
            # else:
            #     self.reinitialize_from_q = reinitialize_from_q_setup

            # Set self.q_factor:
            # if self.q_factor_variable:
            #     self.q_factor = 0.01 / (self.train_size/self.batch_size) * self.num_samp
                # self.q_factor = min(1e-6 * epoch ** 3, 1.)

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
            if generation:
                util.generate_images_from_images(self, test_sample,
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-images.png')
                util.generate_images_from_random(self, random_vector_for_generation, 
                    path=self.file_path + 'generated-images/epoch-'+str(epoch)+'-from-prior.png')
            # Save model
            if epoch % 1 == 0 or epoch == epochs:
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
                if epoch == 20:
                    np.savetxt(self.file_path + 'hmc-points/epoch_20_hmc_points.csv', np.squeeze(self.hmc_points))
                else:
                    np.savetxt(self.file_path + 'hmc-points/hmc_points.csv', np.squeeze(self.hmc_points))
                np.savetxt(self.file_path + 'losses.csv', np.array(losses))
            # Plot HMC points
            # if self.latent_dim == 2 and generation:
            #     if epoch == start_epoch:
            #         old_points = None
            #     old_points = util.plot_hmc_points(self.hmc_points[[0,1000,1999],:], old_points,
            #         path=self.file_path + 'hmc-points/epoch-'+str(epoch)+'.png')

        return






