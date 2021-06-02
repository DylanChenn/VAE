"""
Using keras to reproduce the mnist experiment in the paper.
Using tensorflow backend.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.datasets import mnist

mode = 'tf_opt'  # mode can be one of ['tf_opt', 'self_opt']

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, newshape=(-1, 784))
x_test = np.reshape(x_test, newshape=(-1, 784))
x_input_shape = x_train.shape[1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train[x_train >= 0.5] = 1
x_train[x_train < 0.5] = 0
x_test[x_test >= 0.5] = 1
x_test[x_test < 0.5] = 0
np.random.shuffle(x_train)

h_dim = 500
z_dim = 200
epochs = 50
batch_size = 100
lr = 0.001


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


input_x = tf.placeholder(tf.float32, shape=[None, x_input_shape])
input_z = tf.placeholder(tf.float32, shape=[None, z_dim])

# ===============================Q(Z|X)======================================
q_w1 = tf.Variable(xavier_init([x_input_shape, h_dim]))
q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

q_w2 = tf.Variable(xavier_init([h_dim, h_dim]))
q_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

q_w2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

q_w2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(x):
    hiddenlayer = tf.nn.tanh(tf.matmul(input_x, q_w1) + q_b1)
    # hiddenlayer = tf.nn.tanh(tf.matmul(hiddenlayer, q_w2) + q_b2)
    mu = tf.matmul(hiddenlayer, q_w2_mu) + q_b2_mu
    logvar = tf.matmul(hiddenlayer, q_w2_sigma) + q_b2_sigma
    return mu, logvar


def sampling(mu, logvar):
    epsilon = tf.random_normal(shape=tf.shape(z_mu))
    return mu + tf.exp(logvar / 2) * epsilon


# ===============================P(X|Z)======================================
p_w1 = tf.Variable(xavier_init([z_dim, h_dim]))
p_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

p_w2 = tf.Variable(xavier_init([h_dim, h_dim]))
p_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

p_w3 = tf.Variable(xavier_init([h_dim, x_input_shape]))
p_b3 = tf.Variable(tf.zeros(shape=[x_input_shape]))


def P(z):
    hiddenlayer = tf.nn.tanh(tf.matmul(z, p_w1) + p_b1)
    # hiddenlayer = tf.nn.tanh(tf.matmul(hiddenlayer, p_w2) + p_b2)
    logit = tf.matmul(hiddenlayer, p_w3) + p_b3
    prob = tf.nn.sigmoid(logit)
    return prob, logit


# ================================Loss=======================================
z_mu, z_logvar = Q(input_x)
z_sample = sampling(z_mu, z_logvar)
x_output, x_logit = P(z_sample)
x_generate, _ = P(input_z)

reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=input_x), 1)
kl_loss = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), 1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

# ===============================Training=====================================
x_test_sub = x_test[:15 * 15]
total_count = x_train.shape[0]
batch_count = int(total_count / batch_size - 1) + 1

# using tf optimizer
if mode == 'tf_opt':  # using the optimizer in tensorflow
    # solver = tf.train.GradientDescentOptimizer(lr).minimize(vae_loss)
    solver = tf.train.AdamOptimizer().minimize(vae_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for it in range(batch_count):
            x_batch = x_train[it * batch_size: min((it + 1) * batch_size, total_count)]
            _, loss = sess.run([solver, vae_loss], feed_dict={input_x: x_batch})
            print("Epoch {0}/{4}\t{1}/{2}\tloss:{3}".format(epoch+1, min((it + 1) * batch_size, total_count),
                                                            total_count, loss, epochs))
        x_test_loss = sess.run(vae_loss, feed_dict={input_x: x_test})
        print("=================================================")
        print("Epoch {0}\ttest_loss:{1}".format(epoch + 1, x_test_loss))
        print("=================================================")
    x_test_vae = sess.run(x_output, feed_dict={input_x: x_test_sub})
elif mode == 'self_opt':  # compute the gradients and update the params
    g_qw1, g_qb1, g_qw2, g_qb2, g_qw2mu, g_qb2mu, g_qw2sig, g_qb2sig, g_pw1, g_pb1, g_pw2, g_pb2, g_pw3, g_pb3 = tf.gradients(
        xs=[q_w1, q_b1, q_w2, q_b2, q_w2_mu, q_b2_mu, q_w2_sigma, q_b2_sigma, p_w1, p_b1, p_w2, p_b2,  p_w3, p_b3],
        ys=vae_loss
    )
    n_qw1 = q_w1.assign(q_w1 - lr * g_qw1)
    n_qb1 = q_b1.assign(q_b1 - lr * g_qb1)
    n_qw2 = q_w2.assign(q_w2 - lr * g_qw2)
    n_qb2 = q_b2.assign(q_b2 - lr * g_qb2)
    n_qw2mu = q_w2_mu.assign(q_w2_mu - lr * g_qw2mu)
    n_qb2mu = q_b2_mu.assign(q_b2_mu - lr * g_qb2mu)
    n_qw2sig = q_w2_sigma.assign(q_w2_sigma - lr * g_qw2sig)
    n_qb2sig = q_b2_sigma.assign(q_b2_sigma - lr * g_qb2sig)
    n_pw1 = p_w1.assign(p_w1 - lr * g_pw1)
    n_pb1 = p_b1.assign(p_b1 - lr * g_pb1)
    n_pw2 = p_w2.assign(p_w2 - lr * g_pw2)
    n_pb2 = p_b2.assign(p_b2 - lr * g_pb2)
    n_pw3 = p_w3.assign(p_w3 - lr * g_pw3)
    n_pb3 = p_b3.assign(p_b3 - lr * g_pb3)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for it in range(batch_count):
            x_batch = x_train[it * batch_size: min((it + 1) * batch_size, total_count)]
            _, _, _, _, _, _, _, _, _, _, _, _, _, _, loss = sess.run(
                [n_qw1, n_qb1, n_qw2, n_qb2, n_qw2mu, n_qb2mu, n_qw2sig, n_qb2sig,
                 n_pw1, n_pb1, n_pw2, n_pb2, n_pw3, n_pb3, vae_loss],
                feed_dict={input_x: x_batch}
            )
            print("Epoch {0}/{4}\t{1}/{2}\tloss:{3}".format(epoch + 1, min((it + 1) * batch_size, total_count),
                                                            total_count, loss, epochs))
        x_test_loss = sess.run(vae_loss, feed_dict={input_x: x_test})
        print("=================================================")
        print("Epoch {0}\ttest_loss:{1}".format(epoch + 1, x_test_loss))
        print("=================================================")
    x_test_vae = sess.run(x_output, feed_dict={input_x: x_test_sub})
else:
    raise Exception("Mode can only be 'tf_opt' or 'self_opt'")

# ===========================Testing And Plot=================================
x_test_sub = np.reshape(x_test_sub, newshape=(15, 15, 28, 28))
x_test_vae = np.reshape(x_test_vae, newshape=(15, 15, 28, 28))

if not os.path.exists(r".\Output"):
    os.makedirs(r".\Output")

n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n * 2))
for i in range(0, n):
    for j in range(0, n * 2, 2):
        figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = x_test_sub[i, int(j / 2)]
        figure[i * digit_size:(i + 1) * digit_size, (j + 1) * digit_size:(j + 2) * digit_size] = x_test_vae[
            i, int(j / 2)]
plt.figure(figsize=(10, 10))
plt.title("Latent Variable {0}D".format(z_dim))
plt.imshow(figure, cmap='Greys_r')

plt.savefig(r".\Output\Z_dim={0}_{1}.png".format(z_dim, mode))


n = 32  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

np.random.seed(10)
r_mean = np.zeros(z_dim)
r_cov = np.zeros((z_dim, z_dim))
for i in range(z_dim):
    r_cov[i][i] = 1
random_z = np.random.multivariate_normal(r_mean, r_cov, size=(n * n,))

x_decoded = sess.run(x_generate, feed_dict={input_z: random_z})
for i in range(n):
    for j in range(n):
        digit = x_decoded[i * n + j].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(15, 15))
plt.title("Latent Variable {0}D, Generate from first 2D".format(z_dim))
plt.imshow(figure, cmap='Greys_r')
plt.savefig(r".\Output\GenerateWithZ_dim={0}_{1}.png".format(z_dim, mode))