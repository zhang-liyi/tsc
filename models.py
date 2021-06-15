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

    def __init__(self, v_fam='gaussian', num_dims=2, loc_init=[2.,5.], scale_init=[1.,1.]):
        self.v_fam = v_fam
        self.num_dims = num_dims
        self.target = util.Funnel(num_dims).get_funnel_dist()
        if self.v_fam == 'iaf' or self.v_fam == 'flow':
            self.base_distribution = tfd.Sample(
                tfd.Normal(0., 1.), sample_shape=[num_dims])
            self.made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[20, 20],
                event_shape=(2,),
                activation='elu',
                kernel_initializer=tfk.initializers.GlorotNormal())
            self.make_model()
        else: # v_fam == 'gaussian'
            self.var_loc = tf.Variable(
                tf.zeros(num_dims) + loc_init, 
                name='loc')
            self.var_scale = tfp.util.TransformedVariable(
                tf.zeros(num_dims) + scale_init, 
                tfb.Softplus(),
                name='scale')
            self.q = tfd.MultivariateNormalDiag(
                loc=self.var_loc, 
                scale_diag=self.var_scale)

    def make_model(self):
        x_in = tfkl.Input(shape=(self.num_dims,), dtype=tf.float32) # eps
        x_ = self.made(x_in)
        self.model = tfk.Model(x_in, x_)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)

    def load_model(self, path):
        if self.v_fam == 'iaf' or self.v_fam == 'flow':
            self.model.load_weights(path)
            self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
            self.q = tfd.TransformedDistribution(
                self.base_distribution, 
                self.bij)
        else:
            print('This function is only valid with flow-based VI.')

    def kl_loss(self, q):
        # Define KL loss
        theta = q.sample(1)
        loss = tf.reduce_mean(q.log_prob(theta) - self.target.log_prob(theta))
        return loss

    def train(self, epochs=int(1e5), lr=0.1, save=True):

        self.lambd = self.q.trainable_variables
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        mu = []
        sig = []
        elbo = []

        for epoch in range(epochs):
            if epoch > 10000:
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr/100)
            elif epoch > 200:
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr/10)

            if not (self.v_fam == 'iaf' or self.v_fam == 'flow'):
                mu.append(self.var_loc.numpy())
                sig.append(self.var_scale.numpy())
            
            with tf.GradientTape() as tape:

                loss_value = self.kl_loss(self.q)
                elbo.append(-loss_value)
                
                grads = tape.gradient(loss_value, self.lambd)

            optimizer.apply_gradients(zip(grads, self.lambd))
            
            if epoch % 1 == 0:
                print('epoch', epoch, 'elbo', -loss_value.numpy())

        if save:
            tm = str(datetime.datetime.now())
            tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
            path = 'results/' + 'vi_klqp_' + self.v_fam + '/' + tm_str + '/'
            if not os.path.exists(path): 
                os.makedirs(path)

            elbo = np.array(elbo)
            np.savetxt(path+'elbo.csv', elbo)
            if self.v_fam == 'iaf' or self.v_fam == 'flow':
                self.model.save_weights(path + 'checkpoints/iaf_qp')
            else:
                mu = np.array(mu)
                sig = np.array(sig)
                np.savetxt(path+'mu.csv', mu)
                np.savetxt(path+'sig.csv', sig)
            
            rp = open(path + "run_parameters.txt", "w")
            rp.write('epochs ' + str(epochs) + '\n')
            rp.write('initial learning rate ' + str(lr) + '\n')
            rp.close()



class VI_KLpq:

    def __init__(self, v_fam='gaussian', space='eps', num_dims=2, loc_init=[2.,5.], scale_init=[1.,1.], 
        hmc_e=0.25, hmc_L=4, pt_init=tf.constant([[2,10]], dtype=tf.float32)):
        self.space = space
        self.v_fam = v_fam
        self.num_dims = num_dims
        self.target = util.Funnel(num_dims).get_funnel_dist()
        self.hmc_e = hmc_e
        self.hmc_L = hmc_L
        self.pt_init = pt_init

        if self.v_fam == 'iaf' or self.v_fam == 'flow':
            self.space = 'eps' # it would be meaningless to input v_fam as flow but space as theta
            self.base_distribution = tfd.Sample(
                tfd.Normal(0., 1.), sample_shape=[num_dims])
            self.made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[20, 20],
                event_shape=(2,),
                activation='elu',
                kernel_initializer=tfk.initializers.GlorotNormal())
            self.make_model()
            self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=self.log_hmc_target,
                    step_size=np.float32(self.hmc_e),
                    num_leapfrog_steps=self.hmc_L)
        else: # self.v_fam == 'gaussian'
            self.var_loc = tf.Variable(
                tf.zeros(num_dims) + loc_init, 
                name='loc')
            self.var_scale = tfp.util.TransformedVariable(
                tf.zeros(num_dims) + scale_init, 
                tfb.Softplus(),
                name='scale')
            self.q = tfd.MultivariateNormalDiag(
                loc=self.var_loc, 
                scale_diag=self.var_scale)
            if self.space == 'eps':
                self.bij = tfb.Affine(
                    shift=self.var_loc,
                    scale_diag=self.var_scale)
                self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=self.log_hmc_target,
                    step_size=np.float32(self.hmc_e),
                    num_leapfrog_steps=self.hmc_L)
                self.current_state = self.bij.inverse(pt_init)
            else: # self.space == 'theta'
                self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=self.target.log_prob,
                    step_size=np.float32(self.hmc_e),
                    num_leapfrog_steps=self.hmc_L)
                self.current_state = pt_init

    def make_model(self):
        x_in = tfkl.Input(shape=(self.num_dims,), dtype=tf.float32) # eps
        x_ = self.made(x_in)
        self.model = tfk.Model(x_in, x_)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)
        self.current_state = self.bij.inverse(self.pt_init)

    def load_model(self, path):
        if self.v_fam == 'iaf' or self.v_fam == 'flow':
            self.model.load_weights(path)
            self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
            self.q = tfd.TransformedDistribution(
                self.base_distribution, 
                self.bij)
            self.current_state = self.bij.inverse(self.pt_init)
        else:
            print('This function is only valid with flow-based VI.')

    def log_hmc_target(self, eps):
        # Unnormalized density of p(epsilon | data)
        theta = self.bij.forward(eps)
        part1 = self.target.log_prob(theta)
        part2 = self.bij.forward_log_det_jacobian(eps, 2)
    
        return part1 + part2
    
    def kl_pq_loss(self, theta, q): 
        return -tf.reduce_mean(q.log_prob(theta))

    def train(self, epochs=int(1e5), lr=0.1, num_samp=1, save=True):

        self.lambd = self.q.trainable_variables
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        losses = []
        hmc_points = []
        mu = []
        sig = []
        is_accepted = 0

        for epoch in range(epochs):
            if epoch > 10000:
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr/100)
            elif epoch > 200:
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr/10)
            
            out = tfp.mcmc.sample_chain(num_samp, self.current_state, 
                previous_kernel_results=None, kernel=self.hmc_kernel,
                num_burnin_steps=0, num_steps_between_results=0, 
                trace_fn=(lambda current_state, kernel_results: kernel_results), 
                return_final_kernel_results=False, seed=None, name=None)
            kernel_results = out[1]
            
            if self.space == 'eps':
                eps = out[0]
                theta = self.bij.forward(eps)
            else:
                theta = out[0]
            
            hmc_points.append(theta.numpy())
            is_accepted += np.mean(np.squeeze(kernel_results.is_accepted.numpy()))
            if not (self.v_fam == 'iaf' or self.v_fam == 'flow'):
                mu.append(self.var_loc.numpy())
                sig.append(self.var_scale.numpy())
            
            theta = tf.stop_gradient(theta)
            
            with tf.GradientTape() as tape:
                
                loss_value = self.kl_pq_loss(theta, self.q)
                losses.append(loss_value)
                
                grads = tape.gradient(loss_value, self.lambd)

            optimizer.apply_gradients(zip(grads, self.lambd))
            
            if self.space == 'eps':
                if not (self.v_fam == 'iaf' or self.v_fam == 'flow'):
                    self.bij = tfb.Affine(
                        shift=self.var_loc,
                        scale_diag=self.var_scale)
                self.current_state = self.bij.inverse(tf.gather(theta, num_samp-1))
            else:
                self.current_state = tf.gather(theta, num_samp-1)
            
            if self.v_fam == 'iaf' or self.v_fam == 'flow':
                print('Epoch', epoch, 
                      'Loss', loss_value.numpy(),  
                      'eps', self.current_state.numpy(),
                      'point', tf.gather(theta, num_samp-1).numpy(),
                      'acceptance rate', round(is_accepted/(epoch+1),3))
            else:
                print('Epoch', epoch, 
                      'Loss', np.round(loss_value.numpy(),3),  
                      'point', np.round(tf.gather(theta, num_samp-1).numpy(),3),
                      'mu_param', np.round(self.var_loc.numpy(),3),
                      'sig_param', np.round(self.var_scale.numpy(),3),
                      'acceptance rate', round(is_accepted/(epoch+1),3))

        if save:
            tm = str(datetime.datetime.now())
            tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
            path = 'results/' + 'vi_klpq_' + self.v_fam + '_' + self.space + '/' + tm_str + '/'
            if not os.path.exists(path): 
                os.makedirs(path)

            hmc_points = np.squeeze(np.array(hmc_points))
            losses = np.array(losses)
            np.savetxt(path+'hmc_points.csv', hmc_points)
            np.savetxt(path+'losses.csv', losses)
            if self.v_fam == 'iaf' or self.v_fam == 'flow':
                self.model.save_weights(path + 'checkpoints/iaf_pq')
            else:
                mu = np.array(mu)
                sig = np.array(sig)
                np.savetxt(path+'mu.csv', mu)
                np.savetxt(path+'sig.csv', sig)

            rp = open(path + "run_parameters.txt", "w")
            rp.write('epochs: ' + str(epochs) + '\n')
            rp.write('initial learning rate: ' + str(lr) + '\n')
            rp.write('number of samples during VI training: ' + str(num_samp) + '\n')
            rp.write('HMC step size e: ' + str(self.hmc_e) + '\n')
            rp.write('HMC number of leapfrog steps L: ' + str(self.hmc_L) + '\n')
            rp.write('acceptance rate: ' +str( round(is_accepted/(epochs+1),3) ) + '\n')
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

    def run(self, path=None, save=True):
        if self.space == 'eps':
            if path is not None:
                self.load_model(path)
            self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.log_hmc_target,
                step_size=np.float32(self.hmc_e),
                num_leapfrog_steps=self.hmc_L)
        else:
            self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.target.log_prob,
                step_size=np.float32(self.hmc_e),
                num_leapfrog_steps=self.hmc_L)

        out = tfp.mcmc.sample_chain(
            self.iters, self.current_state, previous_kernel_results=None, kernel=self.hmc_kernel,
            num_burnin_steps=0, num_steps_between_results=0, parallel_iterations=self.chains, 
            trace_fn=(lambda current_state, kernel_results: kernel_results), 
            return_final_kernel_results=False, seed=None, name=None)
        
        accept_rate = np.sum(np.squeeze(out[1].is_accepted.numpy()))/self.iters/self.chains

        if save:
            tm = str(datetime.datetime.now())
            tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
            path = 'results/' + 'hmc_' + self.space + '/' + tm_str + '/'
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



















