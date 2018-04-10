import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio


# Generator Network
def generator(x, scoop_name="Generator", reuse=False):
    with tf.variable_scope(scoop_name, reuse=reuse):
        fc1 = tf.layers.dense(x, units=6 * 6 * 128, activation=tf.nn.tanh)
        # Reshape into 4-D Array as (batch, height, width, channels)
        fc1 = tf.reshape(fc1, shape=[-1, 6, 6, 128])

        # Deconvolution Operation
        trans_conv1 = tf.layers.conv2d_transpose(fc1, 64, 4, strides=2)
        trans_conv2 = tf.layers.conv2d_transpose(trans_conv1, 1, 2, strides=2)

        out = tf.nn.sigmoid(trans_conv2)

        return out


# Discriminator Network
def discriminator(x, scoop_name="Discriminator", reuse=False):
    with tf.variable_scope(scoop_name, reuse=reuse):
        conv1 = tf.layers.conv2d(x, 64, 5, activation=tf.nn.tanh)
        conv1 = tf.layers.average_pooling2d(conv1, pool_size=2, strides=2)

        conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=5, activation=tf.nn.tanh)
        conv2 = tf.layers.average_pooling2d(conv2, pool_size=2, strides=2)

        fc1 = tf.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.tanh)

        out = tf.layers.dense(fc1, 2)

        return out


# Fusion Discriminator Network (cross_class discriminator)
def mlp(x, scoop_name="Discriminator", reuse=False):
    with tf.variable_scope(scoop_name, reuse=reuse):
        x = tf.reshape(x, [-1, 28*28])
        hidden1 = tf.layers.dense(x, units=256, activation=tf.nn.tanh)
        hidden2 = tf.layers.dense(hidden1, units=128, activation=tf.nn.tanh)

        out = tf.layers.dense(hidden2, 2)

        return out


# Fusion Discriminator SVM
def svm(x, scoop_name, reuse=False, alpha=0.01):
    x = tf.reshape(x, [-1, 28*28])
    with tf.variable_scope(scoop_name, reuse=reuse):
        regularizer = tf.contrib.layers.l2_regularizer(scale=alpha)
        da_out = tf.layers.dense(x, 1,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                 kernel_regularizer=regularizer)
    return da_out


# Defining GAN and Operations
def train_operations(noise, image_input, disc_t, gen_t, index):
    gen_scoop = "Generator"+index
    disc_scoop = "Discriminator"+index

    # Build Gen & Disc Net
    gen_sample = generator(noise, scoop_name=gen_scoop)
    disc_real_out = discriminator(image_input, scoop_name=disc_scoop)
    stacked_gan = discriminator(gen_sample, scoop_name=disc_scoop, reuse=True)
    disc_concat = tf.concat([disc_real_out, stacked_gan], axis=0)

    # Loss Definition
    disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=disc_t, logits=disc_concat))
    gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=gen_t, logits=stacked_gan))
    tf.summary.scalar(tensor=disc_loss, name=disc_scoop+" Loss")
    tf.summary.scalar(tensor=gen_loss, name=gen_scoop+" Loss")

    return gen_sample, gen_loss, disc_loss


def cross_class_operations(gen_sample0, gen_sample1, real_image0, real_image1, target_real, target_gen):
    disc_scoop = "Cross_Discriminator"
    disc_class0_real = svm(real_image0, scoop_name=disc_scoop)
    disc_class0_fake = svm(gen_sample0, scoop_name=disc_scoop, reuse=True)
    disc_class1_real = svm(real_image1, scoop_name=disc_scoop, reuse=True)
    disc_class1_fake = svm(gen_sample1, scoop_name=disc_scoop, reuse=True)

    # SVM tries to minimize the classification loss on real images(Training Data)
    # while Gen tries to maximize the loss on generated images(Gen Samples)
    disc_real = tf.concat([disc_class0_real, disc_class1_real], axis=0)
    disc_gen = tf.concat([disc_class0_fake, disc_class1_fake], axis=0)

    # Loss Definition
    disc_loss = tf.reduce_mean(tf.maximum(0., 1. - disc_real * tf.cast(target_real, tf.float32)))
    gen_loss =  tf.reduce_mean(tf.maximum(0., 1. - disc_gen * tf.cast(target_gen, tf.float32)))
    tf.summary.scalar(tensor=gen_loss, name="Generator Cross Loss")
    tf.summary.scalar(tensor=disc_loss, name=disc_scoop + " Loss")

    return gen_loss, disc_loss


def plot_image(sess, gen_sample0, gen_sample1, noise_dim, desired_class, sample_amount, gen_input, idx=None):
    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 5, figsize=(5, 4))
    for i in range(4):
        # Noise input.
        # z = np.random.uniform(-1., 1., size=[4, noise_dim])
        z = np.random.normal(0., 0.3, size=[5, noise_dim])
        if i < 2:
            g = sess.run([gen_sample0], feed_dict={gen_input: z})
        else:
            g = sess.run([gen_sample1], feed_dict={gen_input: z})

        g = np.reshape(g, newshape=(5, 28, 28, 1))
        # Reverse colours for better display
        # g = -1 * (g - 1)
        for j in range(5):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[i][j].imshow(img)

    img_file_name = "./gen_samples/TI_GAN_"+str(desired_class)+"_"+str(sample_amount)
    if idx is not None:
        img_file_name = img_file_name + "_" + str(idx) + ".png"
    else:
        img_file_name += ".png"

    f.show()
    plt.draw()
    plt.savefig(img_file_name)


def gif_plot(desired_class, training_step, sample_num):
    file_prefix = "./gen_samples/TI_GAN_"+str(desired_class)+"_"+str(sample_num)+"_"
    training_step = int(training_step / 1000)
    filenames = [file_prefix+str(i)+".png" for i in range(1, training_step)]
    images = []
    for fn in filenames:
        images.append(imageio.imread(fn))
    kargs = {"duration": 0.75}
    imageio.mimwrite(file_prefix+"demo.gif", images, format="GIF", **kargs)
