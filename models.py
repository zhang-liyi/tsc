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



class VI_KLqp:

    def __init__(self, dataset='funnel', v_fam='gaussian', num_dims=2, 
                 num_samp=1, batch_size=5000, train_size=5000, 
                 loc_init=[2.,5.], scale_init=[1.,1.]):
        self.v_fam = v_fam.lower()
        self.dataset = dataset.lower()
        self.num_dims = num_dims
        self.num_samp = num_samp
        self.batch_size = batch_size
        self.train_size = train_size
        if self.dataset == 'funnel':
            self.num_dims = num_dims
            self.loc_init = [2.,5.]
            self.scale_init = [1.,1.]
        elif self.dataset == 'survey':
            self.num_dims = 123
            self.loc_init = tf.zeros(self.num_dims)
            self.scale_init = tf.ones(self.num_dims)/3
        
        self.likelihood = self.define_likelihood()
        self.prior = self.define_prior()
        self.define_var_dist()
        
        if self.dataset == 'funnel':
            self.trainable_var = self.q.trainable_variables
        elif self.dataset == 'survey':
            self.trainable_var = []
            self.trainable_var.extend(self.q.trainable_variables)
            self.trainable_var.extend([self.gamma_0,
                                       self.gamma, 
                                       self.sigma])
            
    def define_var_dist(self):
        if self.v_fam == 'iaf' or self.v_fam == 'flow':
            self.p_weight = 1
            self.base_distribution = tfd.Sample(
                tfd.Normal(0., 1.), sample_shape=[self.num_dims])
            self.made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[self.num_dims*5, self.num_dims*5],
                event_shape=(self.num_dims,),
                activation='elu',
                kernel_initializer=tfk.initializers.GlorotNormal())
            self.make_model()
        elif self.v_fam == 'gaussian':
            self.p_weight = 1
            self.phi_m = tf.Variable(
                self.loc_init, 
                name='phi_m')
            self.phi_s = tfp.util.TransformedVariable(
                self.scale_init, 
                tfb.Softplus(),
                name='phi_s')
            self.q = tfd.MultivariateNormalDiag(
                loc=self.phi_m, 
                scale_diag=self.phi_s)

    def define_likelihood(self):
        if self.dataset == 'funnel':
            return util.Funnel().get_funnel_dist()
        elif self.dataset == 'survey':
            self.x = tf.zeros((self.batch_size, 128)) # A placeholder to become data
            self.gamma_0 = tf.Variable(0., dtype=tf.float32, name='gamma_0')
            self.gamma = tf.Variable(tf.zeros(5), name='gamma')
            return self.survey_likelihood_lpdf

    def define_prior(self):
        if self.dataset == 'funnel':
            return None
        elif self.dataset == 'survey':
            self.sigma = tf.Variable(
                tf.ones(7), 
                name='sigma')
            return self.survey_prior_lpdf
        
    def survey_likelihood_lpdf(self, alpha):
        splitted_x = tf.split(self.x, [123, 5], axis=1)
        term1 = tf.matmul(splitted_x[0], tf.reshape(alpha, (123, 1)))
        term2 = self.gamma_0
        term3 = tf.matmul(splitted_x[1], tf.reshape(self.gamma, (5, 1)))
        logits = tf.squeeze(term1 + term2 + term3) # has shape (batch_size,)
        return -tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y))
    
    def survey_prior_lpdf(self, alpha):
        splitted_alpha = tf.split(alpha, [50, 6, 4, 5, 8, 30, 20], axis=1)
        prior_lpdf = 0
        for i in range(7):
            prior_lpdf += tf.reduce_sum(util.log_normal_pdf(
                splitted_alpha[i],
                0,
                tf.math.softplus(tf.gather(self.sigma, i))))
        return prior_lpdf
    
    def make_model(self):
        x_in = tfkl.Input(shape=(self.num_dims,), dtype=tf.float32) # eps
        x_ = self.made(x_in)
        self.model = tfk.Model(x_in, x_)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)

    def load_model(self, path):
        self.model.load_weights(path)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)
    
    def kl_loss(self):
        z = self.q.sample(self.num_samp)
        if self.dataset == 'funnel':
            loss = tf.reduce_mean(self.q.log_prob(z) - 
                self.likelihood.log_prob(z))
        elif self.dataset == 'survey':
            loss = tf.reduce_mean(self.q.log_prob(z) - 
                self.likelihood(z) - self.prior(z))
        return loss
    
    def record_data(self, lst):
        if self.dataset == 'funnel':
            if self.v_fam == 'gaussian':
                lst[0].append(self.phi_m.numpy())
                lst[1].append(self.phi_s.numpy())
        elif self.dataset == 'survey':
            if self.v_fam == 'gaussian':
                lst[0].append(self.phi_m.numpy())
                lst[1].append(self.phi_s.numpy())
                lst[2].append(self.gamma_0.numpy())
                lst[3].append(self.gamma.numpy())
                lst[4].append(self.sigma.numpy())
            elif self.v_fam == 'iaf' or self.v_fam == 'flow':
                lst[2].append(self.gamma_0.numpy())
                lst[3].append(self.gamma.numpy())
                lst[4].append(self.sigma.numpy())
        return lst
    
    def train(self, epochs=int(1e5), lr=0.001, save=True, path=None):

        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        phi_m = []
        phi_s = []
        gamma_0 = []
        gamma = []
        sigma = []
        params = [phi_m, phi_s, gamma_0, gamma, sigma]
        params_savenames = ['phi_m.csv', 'phi_s.csv', 'gamma_0.csv', 'gamma.csv', 'sigma.csv']
        elbo = []
        if save:
            tm = str(datetime.datetime.now())
            tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
            if path is None:
                path = 'results/' + self.dataset + '/' + 'vi_klqp_' + self.v_fam + '/' + tm_str + '/'
            else:
                path += self.dataset + '/' + 'vi_klqp_' + self.v_fam + '/' + tm_str + '/'
            if not os.path.exists(path): 
                os.makedirs(path)

        for epoch in range(1, epochs + 1):
            
            params = self.record_data(params)

            with tf.GradientTape() as tape:
                loss_value = self.kl_loss()
                elbo.append(-loss_value.numpy())
                
                grads = tape.gradient(loss_value, self.trainable_var)

            optimizer.apply_gradients(zip(grads, self.trainable_var))
            
            if epoch % 50 == 0 or epoch == 1:
                if self.dataset == 'funnel':
                    if self.v_fam == 'iaf' or self.v_fam == 'flow':
                        print('Epoch', epoch, 
                              'Loss', loss_value.numpy())
                    elif self.v_fam == 'gaussian':
                        print('Epoch', epoch, 
                              'Loss', np.round(loss_value.numpy(),3),
                              'phi_s', np.round(self.phi_s.numpy(),3))
                elif self.dataset == 'survey':
                    if self.v_fam == 'iaf' or self.v_fam == 'flow':
                        print('Epoch', epoch, 
                              'Loss', loss_value.numpy(),  
                              'gamma', np.round(self.gamma.numpy(),3),)
                    elif self.v_fam == 'gaussian':
                        print('Epoch', epoch, 
                              'Loss', np.round(loss_value.numpy(),3),
                              'gamma', np.round(self.gamma.numpy(),3),
                              'phi_s', np.round(np.mean(self.phi_s.numpy()),3))
            if (epoch % 1000 == 0 or epoch == 1) and save:
                np.savetxt(path+'elbo.csv', np.array(elbo))
                if self.v_fam == 'iaf' or self.v_fam == 'flow':
                    self.model.save_weights(path + 'flow_model/model')
                for i in range(len(params)):
                    if len(params[i]) != 0:
                        np.savetxt(path + params_savenames[i], np.array(params[i]))
                if epoch == 1:
                    rp = open(path + "run_parameters.txt", "w")
                    rp.write('dataset: ' + str(self.dataset) + '\n')
                    rp.write('variational family: ' + self.v_fam + '\n')
                    rp.write('number of samples: ' + str(self.num_samp) + '\n')
                    rp.write('epochs: ' + str(epochs) + '\n')
                    rp.write('learning rate: ' + str(lr) + '\n')
                    rp.close()



class VI_KLpq:

    def __init__(self, dataset='funnel', v_fam='gaussian', space='eps', num_dims=2, 
                 num_samp=1, chains=1, hmc_e=0.25, hmc_L=4, 
                 batch_size=5000, train_size=5000, 
                 pt_init=tf.constant([[2,10]], dtype=tf.float32), 
                 loc_init=[2.,5.], scale_init=[1.,1.]):
        self.space = space.lower()
        self.v_fam = v_fam.lower()
        self.dataset = dataset.lower()
        if self.dataset == 'funnel':
            self.num_dims = num_dims
            self.pt_init = tf.tile(pt_init, [chains, 1])
            self.loc_init = loc_init
            self.scale_init = scale_init
        elif self.dataset == 'survey':
            self.num_dims = 123
            self.pt_init = tfd.Sample(tfd.Normal(0,1), self.num_dims).sample(chains)
            self.loc_init = tf.zeros(self.num_dims)
            self.scale_init = tf.ones(self.num_dims)/3
        self.hmc_e = hmc_e
        self.hmc_L = hmc_L
        self.batch_size = batch_size
        self.train_size = train_size
        self.chains = chains
        self.num_samp = num_samp
        
        self.likelihood = self.define_likelihood()
        self.prior = self.define_prior()
        self.define_var_dist()
        self.log_hmc_target = self.define_log_hmc_target()
        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.log_hmc_target,
            step_size=np.float32(self.hmc_e),
            num_leapfrog_steps=self.hmc_L,
            state_gradients_are_stopped=True)
        
        if self.dataset == 'funnel':
            self.trainable_var = self.q.trainable_variables
        elif self.dataset == 'survey':
            self.trainable_var = []
            self.trainable_var.extend(self.q.trainable_variables)
            self.trainable_var.extend([self.gamma_0,
                                       self.gamma, 
                                       self.sigma])
        
    def define_var_dist(self):
        if self.v_fam == 'iaf' or self.v_fam == 'flow':
            self.p_weight = 1
            self.base_distribution = tfd.Sample(
                tfd.Normal(0., 1.), sample_shape=[self.num_dims])
            self.made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[self.num_dims*5, self.num_dims*5],
                event_shape=(self.num_dims,),
                activation='elu',
                kernel_initializer=tfk.initializers.GlorotNormal())
            self.make_model()
        elif self.v_fam == 'gaussian':
            self.p_weight = 1
            self.phi_m = tf.Variable(
                self.loc_init, 
                name='phi_m')
            self.phi_s = tfp.util.TransformedVariable(
                self.scale_init, 
                tfb.Softplus(),
                name='phi_s')
            self.q = tfd.MultivariateNormalDiag(
                loc=self.phi_m, 
                scale_diag=self.phi_s)
            if self.space == 'eps' or self.space == 'warped':
                self.bij = tfb.Affine(
                    shift=self.phi_m,
                    scale_diag=self.phi_s)

    def define_likelihood(self):
        if self.dataset == 'funnel':
            return util.Funnel().get_funnel_dist().log_prob
        elif self.dataset == 'survey':
            self.x = tf.zeros((self.batch_size, 128)) # A placeholder to become data
            self.gamma_0 = tf.Variable(0., dtype=tf.float32, name='gamma_0')
            self.gamma = tf.Variable(tf.zeros(5), name='gamma')
            return self.survey_likelihood_lpdf

    def define_prior(self):
        if self.dataset == 'funnel':
            return None
        elif self.dataset == 'survey':
            self.sigma = tf.Variable(
                tf.ones(7), 
                name='sigma')
            return self.survey_prior_lpdf
        
    def define_log_hmc_target(self):
        if self.space == 'eps' or self.space == 'warped':
            self.current_state = self.bij.inverse(self.pt_init)
            return self.log_hmc_target_warped_space
        else:
            self.current_state = self.pt_init
            return self.log_hmc_target
        
    def survey_likelihood_lpdf(self, alpha):
        splitted_x = tf.split(self.x, [123, 5], axis=1)
        term1 = tf.matmul(splitted_x[0], tf.reshape(alpha, (123, 1)))
        term2 = self.gamma_0
        term3 = tf.matmul(splitted_x[1], tf.reshape(self.gamma, (5, 1)))
        logits = tf.squeeze(term1 + term2 + term3) # has shape (batch_size,)
        return -tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y))
    
    def survey_prior_lpdf(self, alpha):
        # alpha must be 1-by-123
        splitted_alpha = tf.split(alpha, [50, 6, 4, 5, 8, 30, 20], axis=1)
        prior_lpdf = 0
        for i in range(7):
            prior_lpdf += tf.reduce_sum(util.log_normal_pdf(
                splitted_alpha[i],
                0,
                tf.math.softplus(tf.gather(self.sigma, i))))
        return prior_lpdf 
    
    def log_hmc_target_warped_space(self, eps):
        # Unnormalized density p(epsilon | data)
        if self.dataset == 'funnel':
            z = self.bij.forward(eps)
            part1 = self.likelihood(z)
            part2 = self.bij.forward_log_det_jacobian(eps, 1)
            return part1 + part2
        elif self.dataset == 'survey':
            z = self.bij.forward(eps)
            part1 = self.likelihood(z) + self.prior(z)
            part2 = self.bij.forward_log_det_jacobian(eps, 1)
            return part1 + part2
    
    def log_hmc_target(self, z):
        # Unnormalized density p(z | data)
        if self.dataset == 'funnel':
            return self.likelihood(z)
        elif self.dataset == 'survey':
            return self.likelihood(z) + self.prior(z)
    
    def loss(self, z): 
        if self.dataset == 'funnel':
            return -tf.reduce_mean(self.q.log_prob(z)) # Just KL(pq)
        elif self.dataset == 'survey':
            return -tf.reduce_mean(self.q.log_prob(z) + \
                                   self.p_weight*self.likelihood(tf.squeeze(z, axis=0)) + \
                                   self.p_weight*self.prior(tf.squeeze(z, axis=0)))
    
    def make_model(self):
        x_in = tfkl.Input(shape=(self.num_dims,), dtype=tf.float32) # eps
        x_ = self.made(x_in)
        self.model = tfk.Model(x_in, x_)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)

    def load_model(self, path):
        self.model.load_weights(path)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)
        
    def record_data(self, lst):
        if self.dataset == 'funnel':
            if self.v_fam == 'gaussian':
                lst[0].append(self.phi_m.numpy())
                lst[1].append(self.phi_s.numpy())
        elif self.dataset == 'survey':
            if self.v_fam == 'gaussian':
                lst[0].append(self.phi_m.numpy())
                lst[1].append(self.phi_s.numpy())
                lst[2].append(self.gamma_0.numpy())
                lst[3].append(self.gamma.numpy())
                lst[4].append(self.sigma.numpy())
            elif self.v_fam == 'iaf' or self.v_fam == 'flow':
                lst[2].append(self.gamma_0.numpy())
                lst[3].append(self.gamma.numpy())
                lst[4].append(self.sigma.numpy())
        return lst

    def train(self, epochs=int(1e5), lr=0.001, save=True, path=None):

        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        phi_m = []
        phi_s = []
        gamma_0 = []
        gamma = []
        sigma = []
        params = [phi_m, phi_s, gamma_0, gamma, sigma]
        params_savenames = ['phi_m.csv', 'phi_s.csv', 'gamma_0.csv', 'gamma.csv', 'sigma.csv']
        losses = []
        hmc_points = []
        is_accepted = 0
        if save:
            tm = str(datetime.datetime.now())
            tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
            if path is None:
                path = 'results/' + self.dataset + '/' + 'vi_klpq_' + self.v_fam + '/' + tm_str + '/'
            else:
                path += self.dataset + '/' + 'vi_klpq_' + self.v_fam + '/' + tm_str + '/'
            if not os.path.exists(path): 
                os.makedirs(path)

        for epoch in range(1, epochs+1):
            
            out = tfp.mcmc.sample_chain(self.num_samp, self.current_state, 
                previous_kernel_results=None, kernel=self.hmc_kernel,
                num_burnin_steps=0, num_steps_between_results=0, 
                trace_fn=(lambda current_state, kernel_results: kernel_results.is_accepted), 
                parallel_iterations=1000,
                return_final_kernel_results=False, seed=None, name=None)
            results_is_accepted = out[1]
            if self.space == 'eps' or self.space == 'warped':
                eps = tf.gather(out[0], self.num_samp-1)
                z = self.bij.forward(eps)
            else:
                z = tf.gather(out[0], self.num_samp-1)
            z = tf.stop_gradient(z)
            
            params = self.record_data(params)
            hmc_points.append(z.numpy())
            is_accepted += np.mean(np.squeeze(results_is_accepted.numpy()))
            
            with tf.GradientTape() as tape:
                
                loss_value = self.loss(z)
                losses.append(loss_value.numpy())
                
                grads = tape.gradient(loss_value, self.trainable_var)

            optimizer.apply_gradients(zip(grads, self.trainable_var))
            
            if self.space == 'eps' or self.space == 'warped':
                if not (self.v_fam == 'iaf' or self.v_fam == 'flow'):
                    self.bij = tfb.Affine(
                        shift=self.phi_m,
                        scale_diag=self.phi_s)
                self.current_state = self.bij.inverse(z)
            else:
                self.current_state = z
                
            if epoch % 1 == 0:
                if self.dataset == 'funnel':
                    if self.v_fam == 'iaf' or self.v_fam == 'flow':
                        print('Epoch', epoch, 
                              'Loss', loss_value.numpy(), 
                              'eps', np.round(tf.gather(eps, 0).numpy(),3),
                              'point', np.round(tf.gather(z, 0).numpy(),3), 
                              'acceptance rate', round(is_accepted/(epoch+1),3))
                    elif self.v_fam == 'gaussian':
                        print('Epoch', epoch, 
                              'Loss', np.round(loss_value.numpy(),3),
                              'point', np.round(tf.gather(z, 0).numpy(),3), 
                              'phi_s', np.round(self.phi_s.numpy(),3),
                              'acceptance rate', round(is_accepted/(epoch+1),3))
                elif self.dataset == 'survey':
                    if self.v_fam == 'iaf' or self.v_fam == 'flow':
                        print('Epoch', epoch, 
                              'Loss', loss_value.numpy(),  
                              'acceptance rate', round(is_accepted/(epoch+1),3),
                              'gamma', np.round(self.gamma.numpy(),3),)
                    elif self.v_fam == 'gaussian':
                        print('Epoch', epoch, 
                              'Loss', np.round(loss_value.numpy(),3),
                              'acceptance rate', round(is_accepted/(epoch+1),3),
                              'gamma', np.round(self.gamma.numpy(),3),
                              'phi_s', np.round(np.mean(self.phi_s.numpy()),3))
            if (epoch % 1000 == 0 or epoch == 1) and save:
                np.savetxt(path+'losses.csv', np.array(losses))
                np.savetxt(path+'hmc_points.csv', np.reshape(np.squeeze(np.array(hmc_points)), (-1, self.num_dims)))
                if self.v_fam == 'iaf' or self.v_fam == 'flow':
                    self.model.save_weights(path + 'flow_model/model')
                for i in range(len(params)):
                    if len(params[i]) != 0:
                        np.savetxt(path + params_savenames[i], np.array(params[i]))
                if epoch == 1:
                    rp = open(path + "run_parameters.txt", "w")
                    rp.write('dataset: ' + str(self.dataset) + '\n')
                    rp.write('variational family: ' + self.v_fam + '\n')
                    rp.write('epochs: ' + str(epochs) + '\n')
                    rp.write('learning rate: ' + str(lr) + '\n')
                    rp.write('HMC space: ' + self.space + '\n')
                    rp.write('HMC step size e: ' + str(self.hmc_e) + '\n')
                    rp.write('HMC number of leapfrog steps L: ' + str(self.hmc_L) + '\n')
                    rp.write('HMC number of chains: ' + str(self.chains) + '\n')
                    rp.write('acceptance rate: ' +str( round(is_accepted/(epochs),3) ) + '\n')
                    rp.close()



class HMC:
    def __init__(self, space='eps', num_dims=2, iters=int(1e4), chains=1,
        hmc_e=0.25, hmc_L=4):
        self.space = space
        self.iters = iters
        self.chains = chains
        self.hmc_e = hmc_e
        self.hmc_L = hmc_L
        self.target = util.Funnel(num_dims).get_funnel_dist()
        self.base_distribution = tfd.Sample(
                tfd.Normal(0., 1.), sample_shape=[num_dims])
        if self.space == 'eps':
            self.made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[20, 20],
                event_shape=(2,),
                activation='elu',
                kernel_initializer=tfk.initializers.GlorotNormal())
            self.make_model()
        self.current_state = self.base_distribution.sample(self.chains)

    def make_model(self):
        x_in = tfkl.Input(shape=(2,), dtype=tf.float32) # eps
        x_ = self.made(x_in)
        self.model = tfk.Model(x_in, x_)
        self.iaf_bijector_qp = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q_iaf_qp = tfd.TransformedDistribution(
            self.base_distribution, 
            self.iaf_bijector_qp)

    def load_model(self, path):
        if self.space == 'eps':
            self.model.load_weights(path)
            self.iaf_bijector_qp = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
            self.q_iaf_qp = tfd.TransformedDistribution(
                self.base_distribution, 
                self.iaf_bijector_qp)
        else:
            print('This is only valid with HMC on warped space (epsilon-space).')

    def log_hmc_target(self, eps):
        # Unnormalized density, p(epsilon | data)
        # Needs global variables iaf_bijector and target
        theta = self.iaf_bijector_qp.forward(eps)
        part1 = self.target.log_prob(theta)
        part2 = self.iaf_bijector_qp.forward_log_det_jacobian(eps, 1)

        return part1 + part2

    def run(self, load_path=None, save=True, path=None):
        if self.space == 'eps':
            if load_path is not None:
                self.load_model(path)
            self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.log_hmc_target,
                step_size=np.float32(self.hmc_e),
                num_leapfrog_steps=self.hmc_L,
                state_gradients_are_stopped=True)
        else:
            self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.target.log_prob,
                step_size=np.float32(self.hmc_e),
                num_leapfrog_steps=self.hmc_L,
                state_gradients_are_stopped=True)

        out = tfp.mcmc.sample_chain(
            self.iters, self.current_state, previous_kernel_results=None, kernel=self.hmc_kernel,
            num_burnin_steps=0, num_steps_between_results=0, parallel_iterations=self.chains, 
            trace_fn=(lambda current_state, kernel_results: kernel_results.is_accepted), 
            return_final_kernel_results=False, seed=None, name=None)
        
        accept_rate = np.sum(np.squeeze(out[1].numpy()))/self.iters/self.chains

        if save:
            tm = str(datetime.datetime.now())
            tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
            if path is None:
                path = 'results/' + 'hmc_' + self.space + '/' + tm_str + '/'
            else:
                path += 'hmc_' + self.space + '/' + tm_str + '/'
            if not os.path.exists(path): 
                os.makedirs(path)

            if self.space == 'eps':
                out_eps = out[0].numpy() # iters x chains x num_dims
                samps = []
                for i in range(self.chains):
                    samps.append(self.iaf_bijector_qp.forward(out_eps[:,i,:]))
                samps = np.array(samps)
                np.savetxt(path + 'hmc.csv', np.reshape(samps,(-1,2)))
            else:
                samps = out[0].numpy()
                np.savetxt(path + 'hmc.csv', np.reshape(samps,(-1,2)))

            rp = open(path + "run_parameters.txt", "w")
            rp.write('iters: ' + str(self.iters) + '\n')
            rp.write('chains: ' + str(self.chains) + '\n')
            rp.write('HMC step size e: ' + str(self.hmc_e) + '\n')
            rp.write('HMC number of leapfrog steps L: ' + str(self.hmc_L) + '\n')
            rp.write('acceptance rate: ' +str( round(accept_rate,3) ) + '\n')
            rp.close()



















