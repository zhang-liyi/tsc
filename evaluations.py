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



class AIS:

	def __init__(self, init_points, num_anneal_steps=10, hmc_e=0.005, hmc_L=25, num_samp=4):
		self.current_state = init_points
		self.num_anneal_steps = num_anneal_steps
		self.hmc_e = hmc_e 
		self.hmc_L = hmc_L
		self.num_samp = num_samp

		self.j = 2
		self.f_n = lambda x: 0 # to become log target (unnormalized)
		self.f_1 = lambda x: 0 # to become initial f
		self.betas = np.linspace(0, 1, num_anneal_steps, dtype='float32')
		self.log_weights = np.zeros(num_anneal_steps, dtype='float32')

		self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.f_j,
            step_size=self.hmc_e,
            num_leapfrog_steps=self.hmc_L,
            state_gradients_are_stopped=True)

	def f_j(self, z):
		beta = self.betas[self.j-2]
		return (1-beta) * self.f_1(z) + beta * self.f_n(z)

	def run(self):

		for t in range(2, self.num_anneal_steps+1):

			# sample z_t, which is batch_size x latent_dim
			out = tfp.mcmc.sample_chain(
                self.num_samp, self.current_state, 
                previous_kernel_results=None, kernel=self.hmc_kernel,
                num_burnin_steps=0, num_steps_between_results=0, 
                trace_fn=(lambda current_state, kernel_results: kernel_results), 
                return_final_kernel_results=False, seed=None, name=None)

			# update weight
			log_weight = self.log_weights[t-2] - tf.reduce_mean(self.f_j(self.current_state))
			self.j += 1
			log_weight = log_weight + tf.reduce_mean(self.f_j(self.current_state))
			self.current_state = tf.gather(out[0], self.num_samp-1)
			self.log_weights[t-1] = log_weight.numpy()

			print('Anealing step ' + str(t) + '\n', 
				  'Acceptance rate', np.mean(np.squeeze(out[1].is_accepted.numpy()), axis=0))

		return self.log_weights[self.num_anneal_steps-1]




