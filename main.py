import numpy as np
import argparse

import util
import models
import models_vae

def parse_args():
    parser = argparse.ArgumentParser(
                        description='VI and HMC')
    parser.add_argument('--dataset',
                        help='benchmark dataset to use.',
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
    parser.add_argument('--epochs',
                       help='VI total training epochs.',
                       type=int,
                       default=100)
    parser.add_argument('--lr',
                       help='learning rate (of the initial epoch).',
                       type=float,
                       default=0.1)
    parser.add_argument('--batch_size',
                       help='batch size.',
                       type=int,
                       default=100)
    parser.add_argument('--latent_dim',
                        help='VAE latent dimension.',
                        type=int,
                        default=16)
    parser.add_argument('--num_samp',
                       help='number of samples in VI methods.',
                       type=int,
                       default=1)
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
    parser.add_argument('--warm_up', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--load_path',
                       help='Define load_path if you want to load a trained model.',
                       default='na')
    parser.add_argument('--load_epoch',
                       help='Which epoch do you want the loaded model to start trainig at.',
                       type=int,
                       default=1)
    args = parser.parse_args()
    return args

args = parse_args()

if args.method.lower() == 'vi_klqp':
    model = models.VI_KLqp(v_fam=args.v_fam.lower())
    model.target = util.Funnel(2).get_funnel_dist()
    model.train(epochs=args.epochs, lr=args.lr)

if args.method.lower() == 'vi_klpq':
    model = models.VI_KLpq(
        v_fam=args.v_fam.lower(), 
        space=args.space.lower(), 
        hmc_e=args.hmc_e, 
        hmc_L=args.hmc_L)
    model.target = util.Funnel(2).get_funnel_dist()
    model.train(epochs=args.epochs, lr=args.lr, num_samp=args.num_samp)

if args.method.lower() == 'hmc':
    model = models.HMC(space=args.space.lower(), iters=args.iters, chains=args.chains, 
        hmc_e=args.hmc_e, hmc_L=args.hmc_L)
    model.target = util.Funnel(2).get_funnel_dist()
    model.run(path='results/checkpoints/iaf_qp')

if args.method.lower() == 'vae' or args.method.lower() == 'vae_hsc':
    batch_size = args.batch_size
    if args.dataset.lower() == 'mnist':
        train_size = 60000
        test_size = 10000
        train_dataset, test_dataset = util.load_mnist(batch_size)
    random_vector_for_generation = np.genfromtxt('vae_random_vector_' + str(args.latent_dim) + '.csv')
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:16, :, :, :]

    if args.method.lower() == 'vae' and args.v_fam.lower() == 'gaussian':
        model = models_vae.VAE(args.latent_dim, batch_size=batch_size)
        model.train(train_dataset, test_dataset, epochs=args.epochs, lr=args.lr,
            test_sample=test_sample, random_vector_for_generation=random_vector_for_generation)

    if args.method.lower() == 'vae' and (args.v_fam.lower() == 'flow' or args.v_fam.lower() == 'iaf'):
        model = models_vae.VAE_Flow(args.latent_dim, batch_size=batch_size)
        model.train(train_dataset, test_dataset, epochs=args.epochs, lr=args.lr,
            test_sample=test_sample, random_vector_for_generation=random_vector_for_generation)

    if args.method.lower() == 'vae_hsc':
        if args.load_path.lower() == 'na':
            load_path = None
        else:
            load_path = args.load_path
        model = models_vae.VAE_HSC(
            args.latent_dim, 
            num_samp=args.num_samp, 
            hmc_e=args.hmc_e, 
            hmc_L=args.hmc_L,
            batch_size=batch_size,
            train_size=train_size)         
        model.train(train_dataset, test_dataset, epochs=args.epochs, lr=args.lr, stop_idx=train_size, warm_up=args.warm_up,
            test_sample=test_sample, random_vector_for_generation=random_vector_for_generation, generation=True,
            load_path=load_path, load_epoch=args.load_epoch)






