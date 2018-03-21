import os
import dataset
from TI_GAN import *
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Hyper Parameters
sample_amount = 500
# batch_size matters when sample_amount is rather small
batch_size = 8
learning_rate = 2e-4
training_step = 1000*10
display_step = 100

# Network Parameters
image_dim = 784
noise_dim = 64
desired_class = [0, 6]

# Data Feed
# 784 (reshape=True) | 28*28 (reshape=False)
image_reshape = False
data_dir = "/home/ziyi/code/data/"
mnist0 = dataset.read_data_sets(data_dir, target_class=desired_class[0], one_hot=False,
                                reshape=image_reshape, sample_vol=sample_amount)
mnist1 = dataset.read_data_sets(data_dir, target_class=desired_class[1], one_hot=False,
                                reshape=image_reshape, sample_vol=sample_amount)

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
        # batch_x0 = np.reshape(batch_x0, [-1, 28, 28, 1])
        # batch_x1 = np.reshape(batch_x1, [-1, 28, 28, 1])
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
        ops = [
            merged, gen_train0, disc_train0, gen_loss_op0, disc_loss_op0,
            gen_train1, disc_train1, gen_loss_op1, disc_loss_op1,
            cross_gen_loss, cross_disc_loss, cross_gen_train, cross_disc_train
        ]
        summary, _, _, gl0, dl0, _, _, gl1, dl1, cgl, cdl, _, _ = sess.run(ops, feed_dict=feed_dict)

        if idx % display_step == 0:
            history_writer.add_summary(summary, idx)
            print("Step: {:5d}, GL0: {:6f}, DL0: {:6f}, "
                  "GL1: {:6f}, DL1: {:6f}, CGL: {:6f}, CDL: {:6f}".format(idx, gl0, dl0, gl1, dl1, cgl, cdl))
        if (idx+1) % 1000 == 0 and (idx+1) / 1000 > 0:
            d = int((idx+1)/1000)
            plot_image(sess, gen_sample0, gen_sample1, noise_dim, desired_class, sample_amount, gen_input, d)
            # if idx > 2000:
            #     g = sess.run([gen_sample0], feed_dict={gen_input: z})
            #     mnist0.train.concat_batch(g[0])

    history_writer.close()
    gif_plot(desired_class, training_step, sample_amount)
