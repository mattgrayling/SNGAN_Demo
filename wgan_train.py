from wgan import WGAN

gan = WGAN(latent_dims=10, z_lim=0.25, gen_units=100, crit_units=100,
           sn_type='Ic', c_dropout=0.25, g_dropout=0.25, clr=0.00002, glr=0.00002,
           experiment='redshift', n_critic=3, gp_weight=10, redshift=True,
           dataset_path='SNe_Ic_50.csv')
# gan.plot_train_sample()
gan.train(epochs=10, plot_interval=1)
