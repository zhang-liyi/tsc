import numpy as np
import tensorflow as tf
import argparse

import util
import models
import models_vae

def parse_args():
    parser = argparse.ArgumentParser(
                        description='VI and HMC')
    parser.add_argument('--dataset',
                        help='benchmark dataset to use (funnel / survey / mnist).',
                        default='mnist')
    parser.add_argument('--method',
                        help='method, please choose from: {vi_klqp, vi_klpq, hmc, vae, vae_hsc}.',
                        default='vi_klpq')
    parser.add_argument('--v_fam',
                        help='variational family, please choose from: {gaussian, flow, iaf} (the last two are the same here).',
                        default='gaussian')
    parser.add_argument('--space',
                        help='for HSC (VI with KLpq) and HMC only, please choose from: {eps, theta}.',
                        default='eps')
    parser.add_argument('--vae_num_flow',
                       help='Number of flows to stack in VAE. For no flow (i.e. Gaussian), type 0.',
                       type=int,
                       default=0)
    parser.add_argument('--epochs',
                       help='VI total training epochs.',
                       type=int,
                       default=100)
    parser.add_argument('--lr',
                       help='learning rate (of the initial epoch).',
                       type=float,
                       default=0.1)
    parser.add_argument('--decay_rate',
                       help='lr decay rate in inverse time decay.',
                       type=float,
                       default=0.001)
    parser.add_argument('--batch_size',
                       help='batch size.',
                       type=int,
                       default=100)
    parser.add_argument('--latent_dim',
                        help='VAE latent dimension.',
                        type=int,
                        default=2)
    parser.add_argument('--architecture',
                        help='Architecture for encoder and decoder.',
                        default='cnn')
    parser.add_argument('--num_samp',
                       help='number of samples in VI methods.',
                       type=int,
                       default=1)
    parser.add_argument('--likelihood_sigma',
                       help='Sigma when VAE likelihood is diagonal Gaussian.',
                       type=float,
                       default=1.)
    parser.add_argument('--iters',
                       help='HMC iterations.',
                       type=int,
                       default=1000)
    parser.add_argument('--chains',
                       help='HMC chains.',
                       type=int,
                       default=1)
    parser.add_argument('--hmc_e',
                       help='HMC step size.',
                       type=float,
                       default=0.25)
    parser.add_argument('--hmc_L',
                       help='HMC number of leapfrog steps.',
                       type=int,
                       default=4)
    parser.add_argument('--hmc_L_cap',
                       help='Cap on HMC number of leapfrog steps.',
                       type=int,
                       default=4)
    parser.add_argument('--q_factor',
                       help='Factor to weight q - p during training.',
                       type=float,
                       default=1.)
    parser.add_argument('--q_not_train',
                       help='Do not train q for __ epochs.',
                       type=int,
                       default=0)
    parser.add_argument('--target_accept',
                       help='Target acceptance rate.',
                       type=float,
                       default=0.67)
    parser.add_argument('--cis', 
                       help='Number of samples in conditional importance sampling MSC. If 0, no CIS is used.',
                       type=int,
                       default=0)
    parser.add_argument('--hmc_e_differs', 
                       help='Training like Hoffman 2017, where each datapoint has its own step size, if True.',
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--shear', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--reinitialize_from_q', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warm_up', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_encoder', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--natural_gradient', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--stop_idx',
                       help='Index after which training data is not used.',
                       type=int,
                       default=1e8)
    parser.add_argument('--default_path',
                       help='Redefine default_path to save results (if omitted, there will already be a default path).',
                       default='na')
    parser.add_argument('--load_path',
                       help='Define load_path if you want to load a trained model.',
                       default='na')
    parser.add_argument('--load_epoch',
                       help='Which epoch do you want the loaded model to start trainig at.',
                       type=int,
                       default=1)
    parser.add_argument('--random_seed',
                       help='Random seed.',
                       type=int,
                       default=42)
    args = parser.parse_args()
    return args

args = parse_args()

tf.random.set_seed(args.random_seed)

if args.default_path.lower() == 'na':
    path = None
else:
    path = args.default_path
if args.load_path.lower() == 'na':
    load_path = None
else:
    load_path = args.load_path

if args.method.lower() == 'vi_klqp':
    model = models.VI_KLqp(
        dataset=args.dataset, 
        v_fam=args.v_fam.lower(),
        num_samp=args.num_samp)
    model.train(epochs=args.epochs, lr=args.lr, decay_rate=args.decay_rate, save=True, path=path)

if args.method.lower() == 'vi_klpq':
    model = models.VI_KLpq(
        dataset=args.dataset,
        v_fam=args.v_fam.lower(), 
        space=args.space.lower(), 
        num_samp=args.num_samp, 
        chains=args.chains,
        hmc_e=args.hmc_e, 
        hmc_L=args.hmc_L)
    model.train(epochs=args.epochs, lr=args.lr, decay_rate=args.decay_rate, natural_gradient=args.natural_gradient,
        save=True, path=path, load_path=load_path, load_epoch=args.load_epoch)

if args.method.lower() == 'hmc':
    model = models.HMC(
        space=args.space.lower(), 
        iters=args.iters, 
        chains=args.chains, 
        hmc_e=args.hmc_e, 
        hmc_L=args.hmc_L)
    model.run(load_path='results/checkpoints/iaf_qp', save=True, path=path)

if args.method.lower() == 'vae' or args.method.lower() == 'vae_hsc':
    batch_size = args.batch_size
    if args.dataset.lower() == 'mnist':
        train_size = 60000
        test_size = 10000
        train_dataset, test_dataset = util.load_mnist(batch_size)
    elif args.dataset.lower() == 'mnist_dyn':
        train_size = 60000
        test_size = 10000
        train_dataset, test_dataset = util.load_mnist_dyn(batch_size)
    elif args.dataset.lower() == 'fashion_mnist':
        train_size = 60000
        test_size = 10000
        train_dataset, test_dataset = util.load_fashion_mnist(batch_size)
    elif args.dataset.lower() == 'cifar10':
        train_size = 50000
        test_size = 10000
        train_dataset, test_dataset = util.load_cifar10(batch_size)
    random_vector_for_generation = np.genfromtxt(
        'data/vae_random_vector_' + str(args.latent_dim) + '.csv').astype('float32')
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:16, :, :, :]

    if args.method.lower() == 'vae':
        model = models_vae.VAE(
            args.latent_dim,
            num_flow=args.vae_num_flow,
            batch_size=batch_size,
            K=args.num_samp,
            dataset=args.dataset.lower(),
            architecture=args.architecture.lower(),
            likelihood_sigma=args.likelihood_sigma)
        model.train(train_dataset, test_dataset, epochs=args.epochs, lr=args.lr, 
            load_path=load_path, load_epoch=args.load_epoch,
            test_sample=test_sample, random_vector_for_generation=random_vector_for_generation, generation=True)

    if args.method.lower() == 'vae_hsc':
        model = models_vae.VAE_HSC(
            args.latent_dim, 
            num_flow=args.vae_num_flow,
            space=args.space.lower(),
            cis=args.cis,
            dataset=args.dataset.lower(),
            architecture=args.architecture.lower(),
            likelihood_sigma=args.likelihood_sigma,
            num_samp=args.num_samp, 
            chains=args.chains,
            hmc_e=args.hmc_e, 
            hmc_L=args.hmc_L,
            hmc_L_cap=args.hmc_L_cap,
            q_factor=args.q_factor,
            target_accept=args.target_accept,
            batch_size=batch_size,
            train_size=train_size,
            reinitialize_from_q=args.reinitialize_from_q,
            hmc_e_differs=args.hmc_e_differs,
            shear=args.shear)         
        model.train(train_dataset, test_dataset, 
            epochs=args.epochs, lr=args.lr, stop_idx=args.stop_idx, 
            warm_up=args.warm_up, q_not_train=args.q_not_train, train_encoder=args.train_encoder,
            test_sample=test_sample, random_vector_for_generation=random_vector_for_generation, generation=True,
            load_path=load_path, load_epoch=args.load_epoch)






