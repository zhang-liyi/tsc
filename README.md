# hsc

For a typical setting for VAE with HSC, we can run with:
`python main.py --method=vae_hsc --epochs=300 --lr=0.0003 --batch_size=600 --latent_dim=2 --num_samp=1 --hmc_e=0.1 --hmc_L=1`

For a typical setting for HSC on funnel distribution, we can run with:
`python main.py --method=vi_klpq --v_fam=flow --space=eps --epochs=100000 --hmc_e=0.25 --hmc_L=4 --lr=0.03`
