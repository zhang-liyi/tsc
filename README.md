# Transport Score Climbing: Variational Inference using Inclusive KL and Adaptive Neural Transport

Packages used:
* TensorFlow 2.3.0
* TensorFlow-Probability 0.11.1
* TensorFlow Addons

To run:
`python main.py --dataset=[funnel/banana/mnist/mnist_dyn/cifar10] --v_fam=[gaussian/flow] --space=[original/warped] --num_samp=xxx --epochs=xxx --lr=xxx --decay_rate=xxx --hmc_e=xxx --hmc_L=xxx --hmc_L_cap=xxx --cis=xxx --reinitialize_from_q=[true/false] --warm_up=[true/false]`

Some explanations:
* `--dataset=[mnist/mnist_dyn/cifar10]` automatically directs to VAE experiments.
* `--num_samp` refers to number of samples used in VI. It defaults at 1, and when > 1, does IWAE for VAE experiments.
* `--decay_rate` refers to decay rate in inverse time decay learning schedule, which is only used in non-VAE experiments.
* `--cis` defaults at 0. If experiment is VAE, it can be set to > 0 and then the program does conditional importance sampling - Markovian score climbing with cis as number of importance samples.
* `--reinitialize_from_q=[true/false]` refers to whether we reinitialize HMC chain from q in every epoch in VAE experiments. 'True' means TSC; 'false' means NeutraHMC.
