import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

import util

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class AIS:

	def __init__(self, init_points, num_anneal_steps=10, hmc_e=0.01, hmc_L=25, num_samp=4, num_chains=1):
		self.current_state = init_points
		self.num_anneal_steps = num_anneal_steps
		self.hmc_e = hmc_e 
		self.hmc_L = hmc_L
		self.num_samp = num_samp

		self.j = 2
		self.f_n = lambda x: 0 # to become log target (unnormalized)
		self.f_1 = lambda x: 0 # to become initial f

	def run_body(self, current_state, log_weight, log_weight_list, acc_rate_list, hmc_e, j):

		beta = j * 1 / self.num_anneal_steps
		f_j = lambda z: (1-beta) * self.f_1(z) + beta * self.f_n(z)

		hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=f_j,
            step_size=hmc_e,
            num_leapfrog_steps=self.hmc_L,
            state_gradients_are_stopped=True)

		# sample z_t, which is batch_size x latent_dim
		out = tfp.mcmc.sample_chain(
            self.num_samp, current_state, 
            previous_kernel_results=None, kernel=hmc_kernel,
            num_burnin_steps=0, num_steps_between_results=0, 
            trace_fn=(lambda current_state, kernel_results: kernel_results), 
            return_final_kernel_results=False, seed=None, name=None)

		# update weight
		log_weight = tf.gather(log_weight_list, j) - tf.reduce_mean(f_j(current_state))
		j += 1
		beta = j * 1 / self.num_anneal_steps
		f_j = lambda z: (1-beta) * self.f_1(z) + beta * self.f_n(z)
		log_weight = log_weight + tf.reduce_mean(f_j(current_state))
		current_state = tf.gather(out[0], self.num_samp-1)
		log_weight_list = tf.concat([log_weight_list, [log_weight]], axis=0)

		acc_rate = tf.reduce_mean(tf.cast(out[1].is_accepted, tf.float32))
		hmc_e = tf.where(acc_rate < 0.67, hmc_e * 0.98, hmc_e * 1.02)

		tf.print('Anealing step ' + str(j) + '\n', 
			     'HMC step size ' + str(hmc_e.numpy()) +'\n',
				  'Acceptance rate', np.mean(np.squeeze(out[1].is_accepted.numpy())))

		return current_state, log_weight, log_weight_list, acc_rate_list, hmc_e, j


	def run_cond(self, current_state, log_weight, log_weight_list, acc_rate_list, hmc_e, j):
		return j < self.num_anneal_steps

	def run(self):
		z, log_w, log_w_list, acc_rate_list, hmc_e, j = tf.while_loop(self.run_cond, self.run_body, 
												[self.current_state, 0., tf.constant(0., shape=(1,)), 
												 tf.constant(0., shape=(1,)), self.hmc_e, 0],
												 shape_invariants=[tf.TensorShape([]), tf.TensorShape([]),
												                  tf.TensorShape([None]), tf.TensorShape([None]),
												                  tf.TensorShape([]), tf.TensorShape([])])
		return log_w








