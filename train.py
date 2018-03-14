import os
import dataset
from TI_GAN import *
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Hyper Parameters
batch_size = 128
learning_rate = 2e-4
training_step = 1000*3
display_step = 100
data_dir = "/home/ziyi/code/data/"

# Network Parameters
image_dim = 784
noise_dim = 64
desired_calss = [1, 7]

# Data Feed
mnist0 = dataset.read_data_sets(data_dir, target_class=desired_calss[0], one_hot=False)
mnist1 = dataset.read_data_sets(data_dir, target_class=desired_calss[1], one_hot=False)

# Graph Input
gen_input = tf.placeholder(tf.float32, [None, noise_dim])
real_image_input0 = tf.placeholder(tf.float32, [None, 28, 28, 1])
real_image_input1 = tf.placeholder(tf.float32, [None, 28, 28, 1])
# Targets Input
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

disc_target_all = tf.placeholder(tf.int32, shape=[None])
disc_target_gen = tf.placeholder(tf.int32, shape=[None])

# All operations defined on variables
gen_sample0, gen_train0, disc_train0, gen_loss_op0, disc_loss_op0 = \
    train_operations(gen_input, real_image_input0, disc_target, gen_target, index="0")

gen_sample1, gen_train1, disc_train1, gen_loss_op1, disc_loss_op1 = \
    train_operations(gen_input, real_image_input1, disc_target, gen_target, index="1")

cross_gen_loss, cross_disc_loss, cross_gen_train, cross_disc_train = \
    cross_class_operations(gen_sample0, gen_sample1, real_image_input0,
                           real_image_input1, disc_target_all, disc_target_gen)

merged = tf.summary.merge_all()
history_writer = tf.summary.FileWriter("/home/ziyi/code/data/TI_GAN")
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
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
        batch_disc_gen = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
        batch_disc_all = np.concatenate([np.zeros([batch_size*2]), np.ones([batch_size*2])], axis=0)

        feed_dict = {
            gen_input: z, real_image_input0: batch_x0, real_image_input1: batch_x1,
            disc_target: batch_disc_y, gen_target: batch_gen_y,
            disc_target_all: batch_disc_all, disc_target_gen: batch_disc_gen
                     }
        ops = [merged, gen_train0, disc_train0, gen_loss_op0, disc_loss_op0,
               gen_train1, disc_train1, gen_loss_op1, disc_loss_op1,
               cross_gen_loss, cross_disc_loss, cross_gen_train, cross_disc_train]
        summary, _, _, gl0, dl0, _, _, gl1, dl1, cgl, cdl, _, _ = sess.run(ops, feed_dict=feed_dict)

        if idx % display_step == 0:
            history_writer.add_summary(summary, idx)
            print("Step: {:5d}, GL0: {:6f}, DL0: {:6f}, "
                  "GL1: {:6f}, DL1: {:6f}, "
                  "CGL: {:6f}, CDL: {:6f}".format(idx, gl0, dl0, gl1, dl1, cgl, cdl))
    history_writer.close()

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

    f.show()
    plt.draw()
    plt.savefig("./gen_samples/TI_GAN_"+str(desired_calss)+".png")