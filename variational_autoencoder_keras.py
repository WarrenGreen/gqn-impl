'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
from tensorflow.python.keras import backend as K, Input, Model
from tensorflow.python.keras._impl.keras.backend import binary_crossentropy, relu, sigmoid
from tensorflow.python.keras._impl.keras.layers import Lambda, Flatten, Reshape
from tensorflow.python.keras._impl.keras.losses import mse
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.layers.convolutional import Conv2DTranspose, Conv2D
from tensorflow.python.layers.core import Dense
from typing import Tuple

from data_reader import DataReader, Query, Context, _DATASETS


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
image_channels = 3
x_train = np.reshape(x_train, (-1, image_size, image_size, 1))
x_test = np.reshape(x_test, (-1, image_size, image_size, 1))
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

root_path = '/mnt/es0/data/warren/gqn-impl/data'
scene_name = 'rooms_ring_camera'
CONTEXT_SIZE = 4


def data_gen(train_or_test = 'train') -> Tuple:
    data_reader = DataReader(dataset=scene_name, context_size=CONTEXT_SIZE, root=root_path, mode=train_or_test)
    while True:
        data = data_reader.read(batch_size=12)
        query: Query = data[0]
        target_img_batch: np.ndarray = data[1]
        context: Context = query[0]
        query_camera_batch: np.ndarray = query[1]
        context_images: np.ndarray = context[0]
        context_cameras: np.ndarray = context[1]
        yield target_img_batch, target_img_batch


# network parameters
input_shape = (image_size, image_size, image_channels)
intermediate_dim = 784
batch_size = 128
latent_dim = 2
epochs = 50


def get_model(input_shape, intermediate_dim, latent_dim):
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(64, (2,2), activation=relu)(inputs)
    x = Conv2D(64, (3,3), activation=relu)(x)
    x = Conv2D(64, (3,3), activation=relu)(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation=relu)(latent_inputs)
    x = Dense(64*28*28, activation=relu)(x)
    x = Reshape((28, 28, 64))(x)
    x = Conv2DTranspose(64, (3,3), activation=sigmoid, padding='same')(x)
    x = Conv2DTranspose(64, (3,3), activation=sigmoid, padding='same')(x)
    outputs = Conv2D(image_channels, (2, 2), padding='same')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    return vae, encoder, decoder, inputs, z_mean, z_log_var, latent_inputs, outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    vae, encoder, decoder, inputs, z_mean, z_log_var, latent_inputs, outputs = get_model(input_shape, intermediate_dim, latent_dim )
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    # inputs = K.flatten(inputs)
    # outputs = K.flatten(outputs)
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= image_size*image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    #plot_model(vae,
    #           to_file='vae_mlp.png',
    #           show_shapes=True)

    train_data_gen = data_gen()
    test_data_gen =  data_gen('test')
    if args.weights:
        vae.load_weights(args.weights)
    else:
        for i in range(epochs):
            for j in range(60000):
                x, y = next(train_data_gen)
                vae.train_on_batch(x,y)
        ## train the autoencoder
        #vae.fit_generator(train_data_gen,
        #        epochs=epochs,
        #        steps_per_epoch=_DATASETS[scene_name].train_size,
        #        validation_data=test_data_gen,
        #        validation_steps=_DATASETS[scene_name].test_size)
        vae.save_weights('vae_mlp_mnist.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")