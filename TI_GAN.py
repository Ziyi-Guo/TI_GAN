import os
import tensorflow as tf
import dataset
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Hyper Parameters
batch_size = 128
learning_rate = 2e-4
training_step = 1000*10
display_step = 100

# Network Parameters
image_dim = 784
noise_dim = 64
desired_calss = 8

mnist = dataset.read_data_sets("/home/ziyi/code/data/", target_class=desired_calss, one_hot=False)


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

# Graph Input
gen_input = tf.placeholder(tf.float32, [None, noise_dim])
real_image_input = tf.placeholder(tf.float32, [None, 28, 28, 1])

# Build Gen & Disc Net
gen_sample = generator(gen_input)

disc_real_out = discriminator(real_image_input)
disc_fake_out = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real_out, disc_fake_out], axis=0)

stacked_gan = discriminator(gen_sample, reuse=True)

# Targets Input
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

# Loss Definition
disc_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=disc_target, logits=disc_concat))
gen_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=gen_target, logits=stacked_gan))
tf.summary.scalar(tensor=disc_loss_op, name="Discriminator Loss")
tf.summary.scalar(tensor=gen_loss_op, name="Generator Loss")

# Optimizer Definition
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Related variables for two parts
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")

# Trainer in minimizing loss
gen_train = optimizer_gen.minimize(gen_loss_op, var_list=gen_vars)
disc_train = optimizer_gen.minimize(disc_loss_op, var_list=disc_vars)

merged = tf.summary.merge_all()
history_writer = tf.summary.FileWriter("/home/ziyi/code/data/TI_GAN")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for idx in range(training_step):
        # Sample data for Disc and Gen
        batch_x, _ = mnist.train.next_batch(batch_size)
        # batch_x = -1. + batch_x / 2.
        # batch_x = batch_x.eval(session=sess)
        # print(type(batch_x))
        batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
        z = np.random.normal(0., 0.3, size=[batch_size, noise_dim])
        # z = uniform(0., 1., size=[batch_size, noise_dim])

        # Sample labels for Disc
        batch_gen_y = np.ones([batch_size])
        batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)

        feed_dict = {gen_input: z, real_image_input: batch_x,
                     disc_target: batch_disc_y, gen_target: batch_gen_y}
        operations = [merged, gen_train, disc_train, gen_loss_op, disc_loss_op]
        summary, _, _, gl, dl = sess.run(operations, feed_dict=feed_dict)

        if idx % display_step == 0:
            history_writer.add_summary(summary, idx)
            print("Step: {:5d}, Gen_Loss: {:8f}, Disc_Loss: {:8f}".format(
                idx, gl, dl))
    history_writer.close()

    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 5, figsize=(5, 4))
    for i in range(5):
        # Noise input.
        # z = np.random.uniform(-1., 1., size=[4, noise_dim])
        z = np.random.normal(0., 0.3, size=[4, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        # g = [(x + 1.)/2. for x in g]
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        # Reverse colours for better display
        # g = -1 * (g - 1)
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.savefig("./gen_samples/TI_GAN_"+str(desired_calss)+".png")