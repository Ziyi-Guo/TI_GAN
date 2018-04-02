import os
import dataset
from TI_GAN import *
import numpy as np
from sklearn import svm
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Hyper Parameters
sample_amount = 300
# batch_size matters when sample_amount is rather small
batch_size = 8
learning_rate = 2e-4
training_step = 1000*10
display_step = 100

# Network Parameters
image_dim = 784
noise_dim = 64
desired_class = 9

# Data Feed
# 784 (reshape=True) | 28*28 (reshape=False)
image_reshape = False
data_dir = "/home/ziyi/code/data/"
mnist0,mnist1 = dataset.read_data_sets(data_dir, target_class=desired_class, one_hot=False,
                                reshape=image_reshape, sample_vol=sample_amount)

print(mnist1.train.images.shape, mnist1.train.num_examples)

# mnist1 = dataset.read_data_sets(data_dir, target_class=desired_class[1], one_hot=False,
#                                 reshape=image_reshape, sample_vol=sample_amount)

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

clf = svm.SVC(kernel='linear')
# mlp_prev = mlp(real_image_input0, scoop_name="Prev")
# pred = tf.nn.softmax(mlp_prev)
# # mlp_after = mlp(tf.concat([real_image_input0, real_image_input1], axis=0), scoop_name="After")
# mlp_loss = tf.reduce_mean(
#     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=disc_target, logits=mlp_prev))
# mlp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Prev")
# mlp_train = tf.train.AdamOptimizer(learning_rate).minimize(mlp_loss,var_list=mlp_vars)


merged = tf.summary.merge_all()
history_writer = tf.summary.FileWriter("/home/ziyi/code/data/TI_GAN")
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)

    train_images = np.concatenate([mnist0.train.images, mnist1.train.images], 0)
    train_labels = np.concatenate([np.ones(mnist0.train.num_examples), np.zeros(mnist1.train.num_examples)])
    test_num = mnist0.test.num_examples
    test_images = np.concatenate([mnist0.test.images, mnist1.test.images], 0)
    test_labels = np.concatenate([np.ones(test_num), np.zeros(mnist1.test.num_examples)])
    clf.fit(train_images.reshape([-1, 28*28]), train_labels)
    prediction = clf.predict(test_images.reshape([-1, 28*28]))
    print(np.mean(prediction==test_labels)*1.)
    # for idx in range(500):
    #     batch_x0, _ = mnist0.train.next_batch(batch_size)
    #     batch_x1, _ = mnist1.train.next_batch(batch_size)
    #     batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
    #     m_loss, _ = sess.run([mlp_loss, mlp_train],
    #                          feed_dict={real_image_input0: np.concatenate([batch_x0,batch_x1],0),disc_target:batch_disc_y})
    #
    # pred_correct = tf.equal(tf.argmax(pred, 1, output_type=tf.int32), disc_target)
    # accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))
    # print(sess.run([mlp_loss, accuracy],
    #                feed_dict={real_image_input0:test_images, disc_target: test_labels}))

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

        if (idx + 1) % display_step == 0:
            history_writer.add_summary(summary, idx)
            print("Step: {:5d}, GL0: {:6f}, DL0: {:6f}, "
                  "GL1: {:6f}, DL1: {:6f}, CGL: {:6f}, CDL: {:6f}".format(idx + 1, gl0, dl0, gl1, dl1, cgl, cdl))
            if (idx + 1) >= 4000:
                g0, g1 = sess.run([gen_sample0, gen_sample1], feed_dict={gen_input: z})
                mnist0.train.concat_batch(g0)
                # mnist1.train.concat_batch(g1)
        if (idx+1) % 1000 == 0:
            d = int((idx+1)/1000)
            plot_image(sess, gen_sample0, gen_sample1, noise_dim, desired_class, sample_amount, gen_input, d)

    # for idx in range(500):
    #     batch_x0, _ = mnist0.train.next_batch(batch_size)
    #     batch_x1, _ = mnist1.train.next_batch(batch_size)
    #     batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
    #     m_loss, _ = sess.run([mlp_loss, mlp_train],
    #                          feed_dict={real_image_input0: np.concatenate([batch_x0,batch_x1],0),disc_target:batch_disc_y})

    # pred_correct = tf.equal(tf.argmax(pred, 1, output_type=tf.int32), disc_target)
    # accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))
    # test_num = mnist0.test.num_examples
    # test_images = np.concatenate([mnist0.test.images, mnist1.test.images], 0)
    # test_labels = np.concatenate([np.ones(test_num),np.zeros(mnist1.test.num_examples)])
    # print(sess.run([mlp_loss, accuracy],
    #                feed_dict={real_image_input0:test_images, disc_target: test_labels}))

    train_images = np.concatenate([mnist0.train.images, mnist1.train.images], 0)
    train_labels = np.concatenate([np.ones(mnist0.train.num_examples), np.zeros(mnist1.train.num_examples)])
    test_num = mnist0.test.num_examples
    test_images = np.concatenate([mnist0.test.images, mnist1.test.images], 0)
    test_labels = np.concatenate([np.ones(test_num), np.zeros(mnist1.test.num_examples)])
    clf.fit(train_images.reshape([-1, 28 * 28]), train_labels)
    prediction = clf.predict(test_images.reshape([-1, 28 * 28]))
    print(np.mean(prediction == test_labels) * 1.)

    history_writer.close()
    gif_plot(desired_class, training_step, sample_amount)
