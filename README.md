# hsc

Requires:
* Tensorflow 2.x
* Corresponding version of Tensorflow-Probability

For a typical setting of VAE with HSC, can run with:
`python main.py --method=vae_hsc --epochs=300 --lr=0.001 --batch_size=600 --latent_dim=2 --num_samp=1 --hmc_e=0.1 --hmc_L=1`

For a typical setting of HSC on funnel distribution, can run with:
`python main.py --method=vi_klpq --v_fam=flow --space=eps --epochs=100000 --lr=0.03 --hmc_e=0.25 --hmc_L=4`
