# SNGAN
SN Light Curve Generator

This is a demo repo showcasing some of the work on synthetic supernova light curve generation using Generative Adversarial Networks, described in my talk at the Machine Learning for Transients Workshop at the University of Warwick on 11/12th December 2023.

The model architecture for a Wasserstein GAN to generate variable length synthetic supernova photometric time series is contained in the file `wgan.py`. The jupyter notebook `training_example.ipynb` shows a quick example on training a model. A data set of 50 simulated SN Ic light curves is included to test model training. Please note that it may take a large number of training epochs before the model starts generating anything that looks at all realistic.

If you have any questions, feel free to contact me.