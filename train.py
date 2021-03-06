import os
import dataset
from TI_GAN import *
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Hyper Parameters
sample_amount = 200
# batch_size matters when sample_amount is rather small
batch_size = 8
learning_rate = 1e-4
training_step = 1000*6
display_step = 100
alpha = 0.5

# Network Parameters
image_dim = 784
noise_dim = 64
desired_class = [7, 9]

# Data Feed
# 784 (reshape=True) | 28*28 (reshape=False)
image_reshape = False
data_dir = "/home/ziyi/code/data/"
mnist0 = dataset.read_data_sets(data_dir, target_class=desired_class[0], one_hot=False,
                                reshape=image_reshape, sample_vol=sample_amount)
mnist1 = dataset.read_data_sets(data_dir, target_class=desired_class[1], one_hot=False,
                                reshape=image_reshape, sample_vol=sample_amount*10)
test_images = np.concatenate([mnist0.test.images, mnist1.test.images], 0)
test_labels = np.concatenate([np.ones(mnist0.test.num_examples), -np.ones(mnist1.test.num_examples)])
# print(mnist1.train.images.shape, mnist1.train.num_examples)

# Graph Input
gen_input = tf.placeholder(tf.float32, [None, noise_dim])
real_image_input0 = tf.placeholder(tf.float32, [None, 28, 28, 1])
real_image_input1 = tf.placeholder(tf.float32, [None, 28, 28, 1])
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])
disc_target_real = tf.placeholder(tf.int32, shape=[None])
disc_target_gen = tf.placeholder(tf.int32, shape=[None])

# All operations defined on variables
gen_sample0, gen0_loss, disc0_loss = \
    train_operations(gen_input, real_image_input0, disc_target, gen_target, index="0")

gen_sample1, gen1_loss, disc1_loss = \
    train_operations(gen_input, real_image_input1, disc_target, gen_target, index="1")

svm_loss, svm_acc, svm_pred = \
    svm_operations(real_image_input0, real_image_input1, disc_target_real)

cross_gen_loss, cross_disc_loss, real_acc = \
    cross_class_operations(gen_sample0, gen_sample1, real_image_input0,
                           real_image_input1, disc_target_real, disc_target_gen)

gen0_loss = tf.subtract(gen0_loss, alpha * cross_gen_loss)
gen1_loss = tf.subtract(gen1_loss, alpha * cross_gen_loss)

# Varlist of all operations
gen0_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator0")
gen1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator1")
disc0_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator0")
disc1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator1")
cross_disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Cross_Discriminator")
# Optimizer Definition
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# Trainer in minimizing loss
gen0_train = optimizer.minimize(gen0_loss, var_list=gen0_vars)
gen1_train = optimizer.minimize(gen1_loss, var_list=gen1_vars)
svm_train = optimizer.minimize(svm_loss, var_list=cross_disc_vars)
disc0_train = optimizer.minimize(disc0_loss, var_list=disc0_vars)
disc1_train = optimizer.minimize(disc1_loss, var_list=disc1_vars)
disc_cross_train = optimizer.minimize(cross_disc_loss, var_list=cross_disc_vars)

merged = tf.summary.merge_all()
history_writer = tf.summary.FileWriter("/home/ziyi/code/data/TI_GAN")


def get_feed_data(dset0, dset1):
    # Sample data for Disc and Gen
    batch_x0, _ = dset0.train.next_batch(batch_size)
    batch_x1, _ = dset1.train.next_batch(batch_size)
    z = np.random.normal(0., 0.3, size=[batch_size, noise_dim])

    # Sample labels for Disc
    batch_gen_y = np.ones([batch_size])
    batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
    batch_disc_gen = np.concatenate([np.ones([batch_size]), -np.ones([batch_size])], axis=0)
    batch_disc_real = np.concatenate([np.ones([batch_size]), -np.ones([batch_size])], axis=0)

    return batch_x0, batch_x1, z, batch_gen_y, batch_disc_y, batch_disc_gen, batch_disc_real


def train_svm(session, steps, dset0, dset1):
    for idx in range(steps):
        batch_x0, batch_x1, _, _, _, _, batch_disc_real = get_feed_data(dset0, dset1)
        feed_dict = {
            real_image_input0: batch_x0,
            real_image_input1: batch_x1,
            disc_target_real: batch_disc_real
        }
        ops = [svm_train, svm_loss, svm_acc]
        _, l, acc = session.run(ops, feed_dict)

    acc, pred = session.run([svm_acc, svm_pred], feed_dict={real_image_input0: dset0.test.images,
                                                            real_image_input1: dset1.test.images,
                                                            disc_target_real: test_labels})
    pred = pred.reshape(-1, )
    less_acc = np.mean(np.equal(pred[0:dset0.test.num_examples], test_labels[0:dset0.test.num_examples]))
    print("ACC of SVM on test: %f ,less acc : %f  " % (acc, less_acc))
    train_labels = np.concatenate([np.ones(dset0.train.num_examples), -np.ones(dset1.train.num_examples)], 0)
    acc, pred = session.run([svm_acc, svm_pred], feed_dict={real_image_input0: dset0.train.images,
                                                            real_image_input1: dset1.train.images,
                                                            disc_target_real: train_labels})
    pred = pred.reshape(-1,)
    less_acc = np.mean(np.equal(pred[0:dset0.train.num_examples], train_labels[0:dset0.train.num_examples]))

    print("ACC of SVM on train: %f, less acc : %f " % (acc, less_acc))


def train_gen(session, steps, dset0, dset1, base_step=0):
    for idx in range(steps):
        batch_x0, batch_x1, z, batch_gen_y, batch_disc_y, batch_disc_gen, batch_disc_real \
            = get_feed_data(dset0, dset1)

        feed_dict = {
            gen_input: z,
            real_image_input0: batch_x0, real_image_input1: batch_x1,
            disc_target: batch_disc_y, gen_target: batch_gen_y,
            disc_target_real: batch_disc_real,
            disc_target_gen: batch_disc_gen
                     }
        ops = [
            merged, gen0_train, disc0_train, gen0_loss, disc0_loss,
            gen1_train, disc1_train, gen1_loss, disc1_loss,
            cross_gen_loss, cross_disc_loss
        ]
        summary, _, _, gl0, dl0, _, _, gl1, dl1, cgl, cdl = session.run(ops, feed_dict=feed_dict)

        if (idx + 1) % display_step == 0:
            history_writer.add_summary(summary, idx)
            print("Step: {:5d}, GL0: {:6f}, DL0: {:6f}, "
                  "GL1: {:6f}, DL1: {:6f}, CGL: {:6f}, CDL: {:6f}"
                  .format(base_step + idx + 1, gl0, dl0, gl1, dl1, cgl, cdl))

        if (idx + 1) % 1000 == 0:
            plot_image(session, gen_sample0, gen_sample1,
                       noise_dim, desired_class, sample_amount,
                       gen_input, base_step + idx + 1)


def generate(session, vol, dset0, dset1):
    vol_gen = 0
    d0, d1 = None, None
    while vol_gen < vol:
        batch_x0, batch_x1, z, batch_gen_y, batch_disc_y, batch_disc_gen, batch_disc_real = \
            get_feed_data(dset0, dset1)

        feed_dict = {
            gen_input: z,
            real_image_input0: batch_x0, real_image_input1: batch_x1,
            disc_target: batch_disc_y, gen_target: batch_gen_y,
            disc_target_real: batch_disc_real,
            disc_target_gen: batch_disc_gen
        }
        ops = [
            merged,
            gen0_train, disc0_train, gen0_loss, disc0_loss,
            gen1_train, disc1_train, gen1_loss, disc1_loss,
            cross_gen_loss, cross_disc_loss
        ]
        summary, _, _, gl0, dl0, _, _, gl1, dl1, cgl, cdl = session.run(ops, feed_dict=feed_dict)

        if cgl > 0.5:
            g0, g1 = session.run([gen_sample0, gen_sample1], feed_dict={gen_input: z})
            if d0 is None:
                d0, d1 = g0, g1
            else:
                d0, d1 = np.vstack((d0, g0)), np.vstack((d1, g1))
            vol_gen += batch_size
            print("Anti-batch found %d / %d" % (vol_gen, vol))

    dset0.train.concat_batch(g0)
    dset1.train.concat_batch(g1)


init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)

    # Initial run for disc and gen
    train_svm(sess, 3000, mnist0, mnist1)
    train_gen(sess, steps=int(training_step/2), dset0=mnist0, dset1=mnist1, base_step=0)

    b_step = int(training_step/2)
    for i in range(5):
        steps = int(training_step/3)
        generate(sess, 200, mnist0, mnist1)
        train_svm(sess, 1000, mnist0, mnist1)
        train_gen(sess, steps=steps, dset0=mnist0, dset1=mnist1, base_step=b_step)
        b_step += steps

    train_svm(sess, 1000, mnist0, mnist1)
    history_writer.close()
    # gif_plot(desired_class, training_step, sample_amount)
