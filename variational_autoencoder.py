""" Variational Auto-Encoder Example.

Using a variational auto-encoder to generate digits images from noise.
MNIST handwritten digits are used as training examples.

References:
    - Auto-Encoding Variational Bayes The International Conference on Learning
    Representations (ICLR), Banff, 2014. D.P. Kingma, M. Welling
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    - [VAE Paper] https://arxiv.org/abs/1312.6114
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).
    - [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

# Import MNIST data
from tensorflow.contrib.losses import sigmoid_cross_entropy
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.layers.core import Flatten

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# TODO init dataset here

from data_reader import DataReader, Query, Context

root_path = '/mnt/es0/data/warren/gqn-impl/data'
scene_name = 'rooms_ring_camera'
CONTEXT_SIZE = 4
data_reader = DataReader(dataset=scene_name, context_size=CONTEXT_SIZE, root=root_path)

# Parameters
learning_rate = 0.001
num_steps = 30000
batch_size = 64

# Network Parameters
image_dim = 64 # MNIST images are 28x28 pixels, gqn 64x64x3
image_channels = 3
hidden_dim = 512
latent_dim = 2
conv_channels = 64

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# Building the encoder
# context_image = tf.placeholder(tf.float32, shape=[None, CONTEXT_SIZE, image_dim, image_dim, 3])
# context_camera = tf.placeholder(tf.float32, shape=[None, CONTEXT_SIZE, 7])
# query_camera = tf.placeholder(tf.float32, shape=[None, 7])
target_image = tf.placeholder(tf.float32, shape=[None, image_dim, image_dim, image_channels])
# target_image = tf.ones(dtype=tf.float32, shape=[batch_size, image_dim, image_dim, image_channels])
encoder = tf.layers.Conv2D(conv_channels, (2,2), activation=tf.nn.relu)(target_image)
encoder = tf.layers.Conv2D(conv_channels, (3,3), activation=tf.nn.relu)(encoder)
encoder = tf.layers.Conv2D(conv_channels, (3,3), activation=tf.nn.relu)(encoder)
encoder = Flatten()(encoder)

z_mean = tf.layers.Dense(2)(encoder)
z_std = tf.layers.Dense(2)(encoder)

# Sampler: Normal (gaussian) random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
z = z_mean + tf.exp(z_std / 2) * eps
latent_input = tf.placeholder_with_default(z, name='latent_input', shape=[None, 2])
# Building the decoder (with scope to re-use these layers later)
decoder = tf.layers.Dense(image_dim*image_dim, activation=tf.nn.relu)(latent_input)
x = tf.layers.Dense(image_dim*image_dim*3, activation=tf.nn.relu)(decoder)
x = tf.reshape(x, (-1, image_dim, image_dim, 3))
x = tf.layers.Conv2DTranspose(conv_channels, (3,3), activation=tf.nn.relu, padding='same')(x)
x = tf.layers.Conv2DTranspose(conv_channels, (3,3), activation=tf.nn.relu, padding='same')(x)
output = tf.layers.Conv2D(image_channels, (2,2), activation=tf.sigmoid, padding='same')(x)


# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = image_dim*image_dim*image_channels * sigmoid_cross_entropy(x_reconstructed, x_true)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)


loss_op = vae_loss(output, target_image)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        # batch_x, _ = mnist.train.next_batch(batch_size)
        # TODO: read in  batch here
        data = data_reader.read(batch_size=batch_size)
        data = sess.run(data)
        query: Query = data[0]
        target_img_batch: np.ndarray = data[1]
        # context: Context = query[0]
        # query_camera_batch: np.ndarray = query[1]
        # context_images: np.ndarray = context[0]
        # context_cameras: np.ndarray = context[1]
        batch_x = np.reshape(target_img_batch, (-1, image_dim, image_dim, image_channels))

        # Train
        feed_dict = {
            target_image: batch_x}
        _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i, Loss: %f' % (i, l))

    # Testing
    # Generator takes noise as input
    noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])
    ## Rebuild the decoder to create image from noise
    ## Building a manifold of generated digits
    n = 20
    x_axis = np.linspace(-3, 3, n)
    y_axis = np.linspace(-3, 3, n)

    canvas = np.empty((image_dim * n, image_dim * n, image_channels))
    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mu = np.array([[xi, yi]] * batch_size)
            x_mean = sess.run(output, feed_dict={latent_input: z_mu})
            canvas[(n - i - 1) * image_dim:(n - i) * image_dim, j * image_dim:(j + 1) * image_dim] = \
            x_mean[0].reshape(image_dim, image_dim, image_channels)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_axis, y_axis)
    plt.imsave('canvas.png', canvas)
