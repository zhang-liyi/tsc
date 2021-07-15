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



class VAE(tf.keras.Model):
    """variational autoencoder, with the option of adding flow."""

    def __init__(self, latent_dim, num_flow=0, batch_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_flow = num_flow
        self.batch_size = batch_size
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
        self.build_flow()
    
    def build_flow(self):
        self.arnets = []
        self.flows = []
        if self.num_flow > 0:
            for _ in range(self.num_flow):
                arnet = tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                        tfb.AutoregressiveNetwork(
                            params=2,
                            hidden_units=[self.latent_dim*5, self.latent_dim*5],
                            event_shape=self.latent_dim,
                            activation='elu',
                            kernel_initializer=tfk.initializers.GlorotNormal()),
                    ]
                )
                self.arnets.append(arnet)
                self.flows.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(arnet)))
            self.f_tot = tfb.Chain(self.flows)
        else:
            self.f_tot = tfb.Identity()

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
        z0 = self.reparameterize(mean, logvar)
        zt = self.f_tot.forward(z0)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = util.log_normal_pdf(zt, 0., 0.)
        logqz_x = util.log_normal_pdf(z0, mean, logvar) - self.f_tot.forward_log_det_jacobian(z0, 1)
        
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
                for i in range(self.num_flow):
                    self.arnets[i].save_weights(self.file_path +'arnet'+str(i)+'/arnet'+str(i))
            
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
            for i in range(self.num_flow):
                os.makedirs(path+'arnet'+str(i)+'/')
        self.tm_str = tm_str



class VAE_HSC(VAE):
    """variational autoencoder using Hamiltonian score climbing."""
    
    def __init__(self, latent_dim, num_flow=5, 
                 num_samp=1, chains=1, hmc_e=0.25, hmc_L=4, 
                 batch_size=32, train_size=60000, 
                 reinitialize_from_q=True, shear=True):
        super().__init__(latent_dim, num_flow=num_flow, batch_size=batch_size)
        self.num_samp = num_samp
        self.chains = chains
        self.hmc_e = hmc_e
        self.hmc_L = min(max(int(1 / hmc_e), 1), 25) # or self.hmc_L = hmc_L
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
            state_gradients_are_stopped=True)
        self.hmc_points = self.pz.sample(self.chains*self.train_size).numpy()
        self.is_accepted_all_epochs = []
        self.first_epoch = False

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def decode(self, z, apply_sigmoid=False):
        A = self.get_shearing_matrix()
        z = tf.matmul(z, tf.transpose(A))
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def get_shearing_param(self, shear=False):
        if self.shear:
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
            return tf.linalg.set_diag(tf.multiply(self.lower_tri, self.shearing_param), [1]*self.latent_dim)
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
        z0 = self.f_tot.inverse(z)
        logqz_x = util.log_normal_pdf(z0, tf.tile(mean, (self.chains, 1)), tf.tile(logvar, (self.chains, 1))) \
            - self.f_tot.forward_log_det_jacobian(z0, 1)

        return logqz_x
    
    def log_hmc_target(self, z0):
        # Unnormalized density of p(z0 | x)
        zt = self.f_tot.forward(z0)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=tf.tile(self.x_batch, (self.chains, 1, 1, 1)))
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.pz.log_prob(zt)
        logdet = self.f_tot.forward_log_det_jacobian(z0, 1)

        return logpx_z + logpz + logdet

    def reset_hmc_kernel(self):
        smallest_accept = np.min(self.is_accepted)
        average_accept = np.mean(self.is_accepted)
        if smallest_accept < 0.25 or average_accept < 0.67:
            self.hmc_e = self.hmc_e * 0.99
        else:
            self.hmc_e = self.hmc_e * 1.01
        self.hmc_L = min(max(1, int(1 / self.hmc_e)), 25)
        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.log_hmc_target,
            step_size=self.hmc_e,
            num_leapfrog_steps=self.hmc_L,
            state_gradients_are_stopped=True)
        
    def get_current_state(self, idx):  
        if self.reinitialize_from_q:         
            return self.z0_from_q
        else:
            if self.epoch == 1:
                return self.z0_from_q
            else:
                return self.hmc_points[idx:(idx+self.batch_size*self.chains), :]
        
    def modify_current_state(self, idx, zt):
        if idx / self.chains + self.batch_size >= self.train_size:
            self.hmc_points[idx:, :] = zt.numpy()
        else:
            self.hmc_points[idx:(idx+self.batch_size*self.chains), :] = zt.numpy()
    
    def compute_loss(self, x, idx, zt=None):
        mean, logvar = self.encode(x)
        self.z0_from_q = tf.stop_gradient(
            self.reparameterize(
                tf.tile(mean, (self.chains, 1)),
                tf.tile(logvar, (self.chains, 1))))
        
        if zt is None:
            self.x_batch = x
            self.current_state_batch = tf.squeeze(self.get_current_state(idx))
            out = tfp.mcmc.sample_chain(
                self.num_samp, self.current_state_batch, 
                previous_kernel_results=None, kernel=self.hmc_kernel,
                num_burnin_steps=0, num_steps_between_results=0, 
                trace_fn=(lambda current_state, kernel_results: kernel_results), 
                return_final_kernel_results=False, seed=None, name=None)
            kernel_results = out[1]
            z0 = tf.gather(out[0], self.num_samp-1)
            zt = self.f_tot.forward(z0)
            z0 = tf.stop_gradient(z0)
            zt = tf.stop_gradient(zt)
            self.is_accepted = np.mean(np.squeeze(kernel_results.is_accepted.numpy()), axis=0)
            self.is_accepted_all_epochs.append(np.mean(self.is_accepted))
            self.reset_hmc_kernel()
            self.modify_current_state(idx, zt)

        z0 = self.f_tot.inverse(zt)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=tf.tile(x, (self.chains, 1, 1, 1)))
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logqz_x = util.log_normal_pdf(z0, tf.tile(mean, (self.chains, 1)), tf.tile(logvar, (self.chains, 1))) \
            - self.f_tot.forward_log_det_jacobian(z0, 1)

        loss = -tf.reduce_mean(logpx_z + logqz_x)
        
        return loss, zt

    def warm_up_compute_loss(self, x):
        mean, logvar = self.encode(x)
        z0 = self.reparameterize(mean, logvar)
        zt = self.f_tot.forward(z0)
        x_logit = self.decode(zt)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = util.log_normal_pdf(zt, 0., 0.)
        logqz_x = util.log_normal_pdf(z0, mean, logvar) - self.f_tot.forward_log_det_jacobian(z0, 1)
        
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    def train_step(self, x, optimizer, idx, zt=None):
        with tf.GradientTape() as tape:
            loss, zt = self.compute_loss(x, idx, zt=zt)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, zt

    def warm_up_train_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.warm_up_compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def train(self, train_dataset, test_dataset, epochs=10, lr=1e-4, stop_idx=60000, warm_up=False, train_encoder=True,
        test_sample=None, random_vector_for_generation=None, generation=False, load_path=None, load_epoch=1):
        self.write_results_helper('results/vae_hsc/')
        rp = open(self.file_path + "run_parameters.txt", "w")
        rp.write('Latent dimension: ' + str(self.latent_dim) + '\n')
        rp.write('Number of flows: ' + str(self.num_flow) + '\n')
        rp.write('Number of epochs: ' + str(epochs) + '\n')
        rp.write('Learning rate: ' + str(lr) + '\n')
        rp.write('Batch size: ' + str(self.batch_size) + '\n')
        rp.write('HMC number of samples: ' + str(self.num_samp) + '\n')
        rp.write('HMC number of chains: ' + str(self.chains) + '\n')
        rp.write('HMC step size: ' + str(self.hmc_e) + '\n')
        rp.write('HMC number of leapfrog steps: ' + str(self.hmc_L) + '\n')
        rp.write('Use shearing: ' + str(self.shear) + '\n')
        rp.write('Warm up with ELBO training: ' + str(warm_up) + '\n')
        rp.write('Reinitialize HMC from q (in every epoch): ' + str(self.reinitialize_from_q) + '\n')
        if not train_encoder:
            rp.write('Not training encoder during HSC')
        rp.close()

        # Training
        optimizer = tf.keras.optimizers.Adam(lr)
        start_epoch = 1
        hmc_zt = None
        self.decoder.save_weights(self.file_path + 'models/init_decoder/decoder')

        # Load a pretrained encoder (and continue to train it). Meanwhile we want to train a decoder from scratch.
        if load_path is not None:
            self.encoder.load_weights(load_path + 'warmed_up_encoder/encoder')
            for i in range(self.num_flow):
                self.arnets[i].load_weights(load_path +'warmed_up_arnet_'+str(i)+'/arnet'+str(i))
            self.shearing_param = np.genfromtxt(load_path + 'shearing_param.csv')

        # If we don't load, we can do a warm-up and use the warmed-up encoder.
        if warm_up:
            os.makedirs('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_encoder/')
            for i in range(self.num_flow):
                os.makedirs('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_arnet_' + str(i) +'/')
            for epoch in range(1, 21):
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
            self.shearing_param.trainable = False
            # Save model
            self.encoder.save_weights('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_encoder/encoder')
            for i in range(self.num_flow):
                self.arnets[i].save_weights('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/warmed_up_arnet_' + str(i) +'/arnet'+str(i))
            np.savetxt('pretrained/' + str(self.latent_dim) + '/' + self.tm_str + '/shearing_param.csv', self.shearing_param)

        # Main section
        self.decoder.load_weights(self.file_path + 'models/init_decoder/decoder')
        self.encoder.trainable = train_encoder
        for epoch in range(1, epochs + 1):
            print('-- Epoch {} --'.format(epoch))
            idx = 0
            self.epoch = epoch

            for train_x in train_dataset:
                if idx >= stop_idx * self.chains:
                    break

                start_time = datetime.datetime.now()
                loss, hmc_zt = self.train_step(train_x, optimizer, idx, zt=None)
                # for _ in range(1):
                #     loss, hmc_zt_ = self.train_step(train_x, optimizer, idx, zt=hmc_zt)
                idx += self.batch_size * self.chains
                end_time = datetime.datetime.now()
                
                if idx % (self.batch_size*1) == 0:
                    print('Loss', round(loss.numpy(), 3), 
                          'accept rate (avg)', round(self.is_accepted_all_epochs[-1], 3),
                          'step size', np.round(self.hmc_kernel.step_size, 3),
                          'time', end_time-start_time)

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
                np.savetxt(self.file_path+'models/epoch_' + str(epoch) + '_shearing_param/shearing_param.csv', self.shearing_param)
                for i in range(self.num_flow):
                    os.makedirs(self.file_path+'models/epoch_' + str(epoch) + '_arnet_' + str(i) +'/')
                    self.arnets[i].save_weights(self.file_path+'models/epoch_' + str(epoch) + '_arnet_' + str(i) +'/arnet'+str(i))
                # np.savetxt(self.file_path + 'hmc_acceptance.csv', np.squeeze(self.is_accepted_list))
            # np.savetxt(self.file_path + 'hmc-points/hmc_points_' + str(epoch) + '.csv', np.squeeze(self.hmc_points))
            # Plot HMC points
            if self.latent_dim == 2 and generation:
                if epoch == start_epoch:
                    old_points = None
                old_points = util.plot_hmc_points(self.hmc_points[[0,1000,1999],:], old_points,
                    path=self.file_path + 'hmc-points/epoch-'+str(epoch)+'.png')

        return






