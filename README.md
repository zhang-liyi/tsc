# Transport Score Climbing: Variational Inference using Inclusive KL and Adaptive Neural Transport

Packages used:
* TensorFlow 2.3.0
* TensorFlow-Probability 0.11.1
* TensorFlow Addons
* TensorFlow Datasets

To run:
`python main.py --dataset=[funnel/banana/mnist/mnist_dyn/cifar10] --method=[vi_klqp, vi_klpq, vae, vae_mcmc] --v_fam=[gaussian/flow] --space=[original/warped] --num_samp=xxx --epochs=xxx --lr=xxx --decay_rate=xxx --hmc_e=xxx --hmc_L=xxx --hmc_L_cap=xxx --cis=xxx --reinitialize_from_q=[true/false] --warm_up=[true/false]`

Some explanations:
* `--dataset=[mnist/mnist_dyn/cifar10]` must correspond with the `vae_xxx` methods; `--dataset=[funnel/banana]` must correspond with the `vi_klxx` methods.
* `--method`: `vae` includes VAE and IWAE; `vae_mcmc` includes CIS-MSC, NeutraHMC, and TSC. Use `space` argument accordingly.
* `--num_samp` refers to number of samples used in VI. It defaults at 1, and when > 1, does IWAE for VAE experiments.
* `--decay_rate` refers to decay rate in inverse time decay learning schedule, which is only used in non-VAE experiments.
* `--cis` defaults at 0. For VAE-related methods, if `cis` is > 0, then the program does CIS-MSC with `cis` as number of importance samples.
* `--reinitialize_from_q=[true/false]` refers to whether we reinitialize HMC chain from q in every epoch in VAE experiments. 'True' means TSC; 'false' means NeutraHMC.

To run survey dataset, one should use `survey_data.ipynb`.
