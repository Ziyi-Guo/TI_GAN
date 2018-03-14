import tensorflow as tf


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


# Defining GAN and Operations
def train_operations(noise, image_input, disc_t, gen_t, index, lr=2e-4):
    gen_scoop = "Generator"+index
    disc_scoop = "Discriminator"+index

    # Build Gen & Disc Net
    gen_sample = generator(noise, scoop_name=gen_scoop)
    disc_real_out = discriminator(image_input, scoop_name=disc_scoop)
    stacked_gan = discriminator(gen_sample, scoop_name=disc_scoop, reuse=True)
    disc_concat = tf.concat([disc_real_out, stacked_gan], axis=0)


    # Loss Definition
    disc_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=disc_t, logits=disc_concat))
    gen_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=gen_t, logits=stacked_gan))
    tf.summary.scalar(tensor=disc_loss_op, name=disc_scoop+" Loss")
    tf.summary.scalar(tensor=gen_loss_op, name=gen_scoop+" Loss")

    # Optimizer Definition
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # Related variables for two parts
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=gen_scoop)
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=disc_scoop)
    # Trainer in minimizing loss
    gen_train = optimizer.minimize(gen_loss_op, var_list=gen_vars)
    disc_train = optimizer.minimize(disc_loss_op, var_list=disc_vars)

    return gen_sample, gen_train, disc_train, gen_loss_op, disc_loss_op


def cross_class_operations(gen_sample0, gen_sample1, real_image0, real_image1, target_all, target_gen, lr=2e-4/1.5):
    disc_scoop = "Discriminator_Cross"
    disc_class0_real = discriminator(real_image0, scoop_name=disc_scoop)
    disc_class0_fake = discriminator(gen_sample0, scoop_name=disc_scoop, reuse=True)
    disc_class1_real = discriminator(real_image1, scoop_name=disc_scoop, reuse=True)
    disc_class1_fake = discriminator(gen_sample1, scoop_name=disc_scoop, reuse=True)

    disc_all = tf.concat([disc_class0_real, disc_class0_fake, disc_class1_real, disc_class1_fake], axis=0)
    disc_gen = tf.concat([disc_class0_fake, disc_class1_fake], axis=0)

    # Loss Definition
    gen_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_gen, logits=disc_gen))
    disc_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_all, logits=disc_all))
    tf.summary.scalar(tensor=gen_loss_op, name="Generator Cross Loss")
    tf.summary.scalar(tensor=disc_loss_op, name=disc_scoop + " Loss")
    # Optimizer Definition
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # Related variables for two parts
    gen_vars0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator0")
    gen_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator1")
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=disc_scoop)
    # Trainer in minimizing loss
    gen_train = optimizer.minimize(gen_loss_op, var_list=[gen_vars0, gen_vars1])
    disc_train = optimizer.minimize(disc_loss_op, var_list=disc_vars)

    return gen_loss_op, disc_loss_op, gen_train, disc_train
