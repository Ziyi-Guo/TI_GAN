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
training_step = 1000*5
display_step = 100
data_dir = "/home/ziyi/code/data/"

# Network Parameters
image_dim = 784
noise_dim = 64
desired_calss = [0, 6]

mnist0 = dataset.read_data_sets(data_dir, target_class=desired_calss[0], one_hot=False)
mnist1 = dataset.read_data_sets(data_dir, target_class=desired_calss[1], one_hot=False)


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


def train_operations(noise, image_input, disc_t, gen_t, index):
    gen_scoop = "Generator"+index
    disc_scoop = "Discriminator"+index

    # Build Gen & Disc Net
    gen_sample = generator(noise, scoop_name=gen_scoop)
    disc_real_out = discriminator(image_input, scoop_name=disc_scoop)
    disc_fake_out = discriminator(gen_sample, scoop_name=disc_scoop, reuse=True)
    disc_concat = tf.concat([disc_real_out, disc_fake_out], axis=0)

    stacked_gan = discriminator(gen_sample, scoop_name=disc_scoop, reuse=True)

    # Loss Definition
    disc_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=disc_t, logits=disc_concat))
    gen_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=gen_t, logits=stacked_gan))
    tf.summary.scalar(tensor=disc_loss_op, name=disc_scoop+" Loss")
    tf.summary.scalar(tensor=gen_loss_op, name=gen_scoop+" Loss")

    # Optimizer Definition
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # Related variables for two parts
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=gen_scoop)
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=disc_scoop)
    # Trainer in minimizing loss
    gen_train = optimizer.minimize(gen_loss_op, var_list=gen_vars)
    disc_train = optimizer.minimize(disc_loss_op, var_list=disc_vars)

    return gen_sample, gen_train, disc_train, gen_loss_op, disc_loss_op


# Graph Input
gen_input = tf.placeholder(tf.float32, [None, noise_dim])
real_image_input0 = tf.placeholder(tf.float32, [None, 28, 28, 1])
real_image_input1 = tf.placeholder(tf.float32, [None, 28, 28, 1])
# Targets Input
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

gen_sample0, gen_train0, disc_train0, gen_loss_op0, disc_loss_op0 = \
    train_operations(gen_input, real_image_input0, disc_target, gen_target, index="0")

gen_sample1, gen_train1, disc_train1, gen_loss_op1, disc_loss_op1 = \
    train_operations(gen_input, real_image_input1, disc_target, gen_target, index="1")


merged = tf.summary.merge_all()
history_writer = tf.summary.FileWriter("/home/ziyi/code/data/TI_GAN")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for idx in range(training_step):
        # Sample data for Disc and Gen
        batch_x0, _ = mnist0.train.next_batch(batch_size)
        batch_x1, _ = mnist1.train.next_batch(batch_size)
        batch_x0 = np.reshape(batch_x0, [-1, 28, 28, 1])
        batch_x1 = np.reshape(batch_x1, [-1, 28, 28, 1])
        z = np.random.normal(0., 0.3, size=[batch_size, noise_dim])
        # z = uniform(0., 1., size=[batch_size, noise_dim])

        # Sample labels for Disc
        batch_gen_y = np.ones([batch_size])
        batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)

        feed_dict = {gen_input: z, real_image_input0: batch_x0, real_image_input1: batch_x1,
                     disc_target: batch_disc_y, gen_target: batch_gen_y}
        ops = [merged, gen_train0, disc_train0, gen_loss_op0, disc_loss_op0,
               gen_train1, disc_train1, gen_loss_op1, disc_loss_op1]
        summary, _, _, gl0, dl0, _, _, gl1, dl1 = sess.run(ops, feed_dict=feed_dict)

        if idx % display_step == 0:
            history_writer.add_summary(summary, idx)
            print("Step: {:5d}, Gen_Loss0: {:8f}, Disc_Loss0: {:8f}, "
                  "Gen_Loss1: {:8f}, Disc_Loss1: {:8f}".format(idx, gl0, dl0, gl1, dl1))
    history_writer.close()

    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 5, figsize=(5, 4))
    for i in range(5):
        # Noise input.
        # z = np.random.uniform(-1., 1., size=[4, noise_dim])
        z = np.random.normal(0., 0.3, size=[4, noise_dim])
        if i<2:
            g = sess.run([gen_sample0], feed_dict={gen_input: z})
        else:
            g = sess.run([gen_sample1], feed_dict={gen_input: z})

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