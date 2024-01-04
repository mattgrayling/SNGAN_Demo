import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import scipy.optimize as spopt
import scipy.stats as stats
import pickle
import numpy.random as rand
import matplotlib as mpl
from matplotlib import rc
import time

rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import Distance
import astropy.units as u
from tqdm import tqdm, trange
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers as opt
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Reshape, BatchNormalization, Activation, Concatenate
import tensorflow.keras.backend as K
import george
from george.kernels import Matern32Kernel
import time
import corner
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
#import umap
#from prdc import compute_prdc

tf.keras.utils.disable_interactive_logging()

# tf.config.run_functions_eagerly(True)

plt.rcParams.update({'font.size': 26})
pd.options.mode.chained_assignment = None

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class WGANModel(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dims,
            discriminator_extra_steps=1,
            gp_weight=1.0,
    ):
        super(WGANModel, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dims = latent_dims
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    @tf.function(input_signature=[tf.TensorSpec([None, None, 10], tf.float32)])
    def call(self, inputs):
        x = self.generator(inputs)
        return x

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGANModel, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_data, fake_data):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_data - real_data
        interpolated = real_data + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(1e-12 + tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, X):
        # Get the batch size
        if isinstance(X, tuple):
            X = X[0]
        batch_size = tf.shape(X)[0]
        timesteps = tf.shape(X)[1]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            noise = tf.random.normal((batch_size, self.latent_dims))
            noise = tf.reshape(noise, (batch_size, 1, self.latent_dims))
            noise = tf.repeat(noise, timesteps, 1)
            self.__call__(noise)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(noise, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(X, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, X, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        # Train the generator
        # Get the latent vector
        noise = tf.random.normal((batch_size, self.latent_dims))
        noise = tf.reshape(noise, (batch_size, 1, self.latent_dims))
        noise = tf.repeat(noise, timesteps, 1)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_data = self.generator(noise, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_data, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


class WGAN:
    """
    Wasserstein GAN implementation for supernova light curve generation
    """

    def __init__(self, latent_dims=100, clr=0.0005, glr=0.0005, GP=True, dataset_path=None, z_lim=None, g_dropout=0.5,
                 c_dropout=0.5, redshift=False, gen_units=100, crit_units=100, sn_type='II', n_critic=1, gp_weight=10):
        """
        :param latent_dims: int, number of latent dimensions to draw random seed for generator from
        :param clr: float, initial learning rate to use for critic
        :param glr: float, initial learning rate to use for generator
        :param GP: Boolean, specifies whether to Gaussian Process interpolate training light curves
        :param z_lim: float, upper redshift limit for sample
        :param g_dropout: float, dropout fraction to use in generator model
        :param c_dropout: float, dropout fraction to use in critic model
        :param gen_units: int, number of GRU units in each layer of generator model
        :param crit_units: int, number of GRU units in each layer of generator model
        :param sn_type: string, SN class to look at
        """
        self.latent_dims = latent_dims
        self.clr = clr
        self.glr = glr
        self.GP = GP
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        self.z_lim = z_lim
        self.g_dropout = g_dropout
        self.c_dropout = c_dropout
        self.gen_units = gen_units
        self.crit_units = crit_units
        self.sn_type = sn_type
        self.redshift = redshift
        # WGAN Paper guidance-----------------------------
        self.n_critic = n_critic
        self.c_optimizer = opt.Adam(lr=self.clr, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = opt.Adam(lr=self.glr, beta_1=0.5, beta_2=0.9)
        self.gp_weight = gp_weight
        # ------------------------------------------------
        # --

        type_dict = {}

        type_dict[1] = [834, 825, 840, 845, 819, 851]  # IIn
        type_dict[2] = [830, 829, 822, 847, 865, 849, 850, 848, 832, 859, 804]  # IIb
        type_dict[0] = [864, 866, 835, 844, 837, 838, 855, 863, 843, 861, 860, 858, 857, 856, 852, 839,
                        801, 802, 867, 824, 808, 811, 831, 817]
        type_dict[3] = [833, 807, 854, 813, 815, 816, 803, 827, 821, 842, 841, 828, 818]  # Ib
        type_dict[4] = [846, 805, 862, 823, 814, 812, 810]  # Ic
        type_dict[5] = [826, 806, 853, 836, 820, 809]  # Ic-BL

        self.type_dict = {}
        for key, vals in type_dict.items():
            for val in vals:
                self.type_dict[val] = key
        self.class_label_decoder = {0.0: 'II', 1.0: 'IIn', 2.0: 'IIb', 3.0: 'Ib', 4.0: 'Ic', 5.0: 'Ic-BL'}
        self.class_label_encoder = {val: key for key, val in self.class_label_decoder.items()}
        # --
        self.n_output = 12
        self.name = f'WGAN_DES_sim_{self.sn_type}_CCSNe_clr{self.clr}_glr{self.glr}_ld{self.latent_dims}' \
                    f'_GP{self.GP}_zlim{self.z_lim}_gN{self.gen_units}_cN{self.crit_units}' \
                    f'_gd{self.g_dropout}_cd{self.c_dropout}' \
                    f'_ncrit{self.n_critic}_gpw{self.gp_weight}_redshift{self.redshift}'

        if not os.path.exists(os.path.join('Data', 'Models', 'Weights')):
            os.mkdir(os.path.join('Data', 'Models', 'Weights'))
        if not os.path.exists(os.path.join('Data', 'Models', 'Plots')):
            os.mkdir(os.path.join('Data', 'Models', 'Plots'))
        if not os.path.exists(os.path.join('Data', 'Models', 'LCs')):
            os.mkdir(os.path.join('Data', 'Models', 'LCs'))
        self.weight_root = os.path.join('Data', 'Models', 'Weights', self.name)
        self.plot_root = os.path.join('Data', 'Models', 'Plots', self.name)
        self.lc_root = os.path.join('Data', 'Models', 'LCs', self.name)
        if not os.path.exists(self.weight_root):
            os.mkdir(self.weight_root)
        if not os.path.exists(self.plot_root):
            os.mkdir(self.plot_root)
        if not os.path.exists(self.lc_root):
            os.mkdir(self.lc_root)
        data = pd.read_csv(os.path.join('Data', 'Datasets', dataset_path))
        data = data[data.sn_type == self.class_label_encoder[sn_type]]
        # data = data[data.sn.isin(data.sn.unique()[:50])]
        min_t, max_t = data[['g_t', 'r_t', 'i_t', 'z_t']].values.min(), data[['g_t', 'r_t', 'i_t', 'z_t']].values.max()
        min_mag, max_mag = data[['g', 'r', 'i', 'z']].values.min(), data[['g', 'r', 'i', 'z']].values.max()
        min_err, max_err = data[['g_err', 'r_err', 'i_err', 'z_err']].values.min(), \
                           data[['g_err', 'r_err', 'i_err', 'z_err']].values.max()
        min_z, max_z = np.min(data[['redshift']].values), np.max(data[['redshift']].values)
        data[['g_t', 'r_t', 'i_t', 'z_t']] = 2 * ((data[['g_t', 'r_t', 'i_t', 'z_t']].values - min_t) / (max_t - min_t) - 0.5)
        data[['g', 'r', 'i', 'z']] = 2 * ((data[['g', 'r', 'i', 'z']].values - min_mag) / (max_mag - min_mag) - 0.5)
        data[['g_err', 'r_err', 'i_err', 'z_err']] = 2 * ((data[['g_err', 'r_err', 'i_err', 'z_err']].values - min_err) / (max_err - min_err) - 0.5)
        data['redshift'] = 2 * ((data['redshift'].values - min_z) / (max_z - min_z) - 0.5)
        self.train_df = data
        self.scaling_factors = (min_t, max_t, min_mag, max_mag, min_err, max_err, min_z, max_z)
        self.train_df = self.train_df[self.train_df.sn_type == self.class_label_encoder[sn_type]]
        self.wgan_dir = os.path.join(self.weight_root, 'model_weights')
        print(self.train_df.sn.unique().shape)

        '''
        # Optimizer

        # Build discriminator
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss, optimizer=self.c_optimizer)
        print(self.critic.summary())

        # Build generator
        self.generator = self.build_generator()
        print(self.generator.summary())

        # Build combined model

        i = Input(shape=(None, self.latent_dims))
        lcs = self.generator(i)

        self.critic.trainable = False

        valid = self.critic(lcs)

        self.combined = Model(i, valid)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=self.g_optimizer)
        print(self.combined.summary())'''
        self.wgan = WGANModel(self.build_critic(), self.build_generator(), self.latent_dims,
                              gp_weight=self.gp_weight, discriminator_extra_steps=self.n_critic)
        self.wgan.compile(d_optimizer=self.c_optimizer, g_optimizer=self.g_optimizer,
                          g_loss_fn=self.generator_loss, d_loss_fn=self.discriminator_loss)

    def wasserstein_loss(self, y_true, y_pred):
        """
        Loss function for Wasserstein GAN
        :param y_true: True labels of data
        :param y_pred: Output of critic model
        :return: Loss
        """
        return K.mean(y_true * y_pred)

    def build_generator(self):
        """
        Builds generator model
        :return: model, keras Model object for generator
        """
        input = Input(shape=(None, self.latent_dims))
        gru1 = GRU(self.gen_units, activation='tanh', return_sequences=True)(input)
        dr1 = Dropout(self.g_dropout)(gru1)
        gru2 = GRU(self.gen_units, activation='tanh', return_sequences=True)(dr1)
        dr2 = Dropout(self.g_dropout)(gru2)
        output = GRU(self.n_output, return_sequences=True, activation='tanh')(dr2)
        model = Model(input, output)
        return model

    def build_critic(self):
        """
        Builds critic model
        :return: model, keras Model object for critic
        """
        input = Input(shape=(None, self.n_output))
        gru1 = GRU(self.crit_units, return_sequences=True)(input)
        dr1 = Dropout(self.c_dropout)(gru1)
        gru2 = GRU(self.crit_units)(dr1)
        dr2 = Dropout(self.c_dropout)(gru2)
        output = Dense(1, activation=None)(dr2)
        model = Model(input, output)
        return model

    def plot_train_sample(self):
        """
        Generates light curve plots for training sample
        """
        print('Generating plots for training sample...')
        if not os.path.exists(os.path.join(self.plot_root, 'Training_sample')):
            os.mkdir(os.path.join(self.plot_root, 'Training_sample'))
        for sn in tqdm(self.train_df.sn.unique()):
            sndf = self.train_df[self.train_df.sn == sn]
            plt.figure(figsize=(12, 8))
            for b_ind, band in enumerate(['g', 'r', 'i', 'z']):
                ax = plt.subplot(2, 2, b_ind + 1)
                ax.scatter(sndf[f'{band}_t'], sndf[band], label=band)
                ax.legend()
            plt.savefig(os.path.join(self.plot_root, 'Training_sample', f'{sn}.jpg'))
            plt.close('all')

    @staticmethod
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    @staticmethod
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

    def train(self, epochs=100, batch_size=1, plot_interval=None):
        """
        Trains generator and critic
        :param epochs: int, number of epochs to run training for
        :param batch_size: int, size of each batch (currently only works for size of 1)
        :param plot_interval: int, number of epochs between showing examples plots
        """
        print('Starting training...')
        if not os.path.exists(os.path.join(self.plot_root, 'Train_plots')):
            os.mkdir(os.path.join(self.plot_root, 'Train_plots'))
        rng = np.random.default_rng(123)

        sne = self.train_df.sn.unique()

        n_batches = int(len(sne) / batch_size)

        current_epoch = 0
        if not os.path.exists(self.wgan_dir):
            os.mkdir(self.wgan_dir)

        real = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        self.train_df.to_csv('input_data.csv')

        epoch_g_losses, epoch_d_losses = [], []

        for epoch in range(current_epoch, epochs):
            rng.shuffle(sne)
            g_losses, d_losses, real_predictions, fake_predictions = [], [], [], []
            t = trange(n_batches)
            for batch in t:
                # Select real data
                sn = sne[batch]
                sndf = self.train_df[self.train_df.sn == sn]
                sndf[['g_t', 'r_t', 'i_t', 'z_t']] = 2 * (sndf[['g_t', 'r_t', 'i_t', 'z_t']] - 0.5)
                sndf[['g', 'r', 'i', 'z']] = 2 * (sndf[['g', 'r', 'i', 'z']] - 0.5)
                sndf[['g_err', 'r_err', 'i_err', 'z_err']] = 2 * (sndf[['g_err', 'r_err', 'i_err', 'z_err']] - 0.5)
                X = sndf[['g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                                      'r_err', 'i_err', 'z_err']].values

                sn_type = X[0, -1]
                X = X.reshape((1, *X.shape))
                d_loss, g_loss = self.wgan.train_on_batch(X)

                if np.count_nonzero(np.isnan(X)) > 0:
                    continue

                # noise = rand.uniform(-1, 1, size=(batch_size, self.latent_dims))
                noise = rand.normal(size=(batch_size, self.latent_dims))
                noise = np.reshape(noise, (batch_size, 1, self.latent_dims))
                noise = np.repeat(noise, X.shape[1], 1)

                test_gen_lcs = self.wgan.generator.predict(noise)
                if np.count_nonzero(np.isnan(test_gen_lcs)) > 0:
                    raise ValueError('NaN generated, check how this happened')

                d_losses.append(d_loss)
                g_losses.append(g_loss)
                t.set_description(f'g_loss={np.around(np.mean(g_losses), 5)},'
                                  f' d_loss={np.around(np.mean(d_losses), 5)}')
                t.refresh()
            full_g_loss = np.mean(g_losses)
            full_d_loss = np.mean(d_losses)
            print(f'{epoch + 1}/{epochs} g_loss={full_g_loss}, d_loss={full_d_loss}')  # , '
            # f'Real prediction: {np.mean(real_predictions)} +- {np.std(real_predictions)}, '
            # f'Fake prediction: {np.mean(fake_predictions)} +- {np.std(fake_predictions)}')
            # f' Ranges: x [{np.min(gen_lcs[:, :, 0])}, {np.max(gen_lcs[:, :, 0])}], '
            # f'y [{np.min(gen_lcs[:, :, 1])}, {np.max(gen_lcs[:, :, 1])}]')

            if (epoch + 1) % plot_interval == 0:
                self.wgan.save(os.path.join(self.wgan_dir, f'{epoch + 1}.tf'))
                plot_test = test_gen_lcs[0, :, :]
                fig = plt.figure(figsize=(12, 8))
                x = plot_test[:, 0]
                X = X.reshape((*X.shape[1:],))
                for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                    ax = fig.add_subplot(2, 2, f_ind + 1)
                    x, y, y_err = plot_test[:, f_ind], plot_test[:, f_ind + 4], plot_test[:, f_ind + 8]
                    ax.scatter(x, y) #, yerr=y_err, label=f, fmt='x')
                    # ax.errorbar(X[:, f_ind], X[:, f_ind + 4], yerr=X[:, f_ind + 8], fmt='x')
                    ax.legend()
                plt.suptitle(f'Epoch {epoch + 1}/{epochs}')  #: Type {self.class_label_dict[sn_type]}')
                plt.savefig(os.path.join(self.plot_root, 'Train_plots', f'{epoch + 1}.png'))
                plt.show()

    def lc_plot(self, col, row, scale=4, epoch=1000, timesteps=12, file_format='png'):
        if not os.path.exists(os.path.join(self.plot_root, 'Tile_Plots')):
            os.mkdir(os.path.join(self.plot_root, 'Tile_Plots'))
        if not os.path.exists(os.path.join(self.plot_root, 'Tile_Plots', str(epoch))):
            os.mkdir(os.path.join(self.plot_root, 'Tile_Plots', str(epoch)))
        n = row * col
        colour_dict = {'g': 'g', 'r': 'r', 'i': 'b', 'z': 'k'}
        self.wgan = keras.models.load_model(os.path.join(self.wgan_dir, f'{epoch}.tf'))
        fig, axs = plt.subplots(row, col, figsize=(scale * col, scale * row))
        for ind in range(n):
            ax = axs.flatten()[ind]
            sn = np.random.choice(self.train_df.sn.unique())
            sndf = self.train_df[self.train_df.sn == sn]
            sndf = sndf[['sn', 'g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                         'r_err', 'i_err', 'z_err']]
            if timesteps is None:
                n_steps = np.min([sndf.shape[0], 18])
                # elif n_steps > 15:
                #     n_steps -= 5
            else:
                n_steps = timesteps

            noise = rand.normal(size=(1, self.latent_dims))
            noise = np.reshape(noise, (1, 1, self.latent_dims))
            noise = np.repeat(noise, n_steps, 1)
            gen_lcs = self.wgan.generator.predict(noise)
            X = gen_lcs[0, :, :]
            if X.shape[1] == 13:  # Drop redshift column for plotting
                X = X[:, :-1]
            gen_sndf = pd.DataFrame(X, columns=sndf.columns[1:])
            # gen_sndf = sndf.copy()
            gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] = (gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] + 1) / 2
            gen_sndf[['g', 'r', 'i', 'z']] = (gen_sndf[['g', 'r', 'i', 'z']] + 1) / 2
            gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] = (gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] + 1) \
                                                             / 2
            gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] = gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                    self.scaling_factors[5] - self.scaling_factors[4]) + self.scaling_factors[4]
            gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] = gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] * \
                                                     (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                                     self.scaling_factors[0]
            gen_sndf[['g', 'r', 'i', 'z']] = gen_sndf[['g', 'r', 'i', 'z']] * (
                    self.scaling_factors[3] - self.scaling_factors[2]) + self.scaling_factors[2]
            gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] = gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                    self.scaling_factors[3] - self.scaling_factors[2]) # * -1
            t_max = gen_sndf[gen_sndf.g == gen_sndf.g.min()].g_t.values[0]
            gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] -= t_max
            for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                ax.errorbar(gen_sndf[f'{f}_t'], gen_sndf[f], yerr=gen_sndf[f'{f}_err'],
                            color=colour_dict[f], fmt='x', label=f)
            # if ind >= n - col:
            #     ax.set_xlabel('Phase')
            # if ind % col == 0:
            #     ax.set_ylabel('Apparent Magnitude')
            ax.invert_yaxis()
        ax0 = fig.add_subplot(111, frameon=False)
        ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                        labelbottom=False, labelleft=False, labeltop=False)
        ax0.set_xlabel('Phase', labelpad=40)
        ax0.set_ylabel('Apparent Magnitude', labelpad=40)
        axs.flatten()[int(np.floor(col / 2))].legend(bbox_to_anchor=(0.5, 1.35), loc='upper center', ncol=4)
        plt.savefig(os.path.join(self.plot_root, 'Tile_Plots', str(epoch), f'{row}x{col}.{file_format}'),
                    bbox_inches='tight')
        plt.show()
