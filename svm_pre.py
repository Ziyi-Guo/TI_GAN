import dataset
from sklearn import svm
import numpy as np

# Hyper Parameters
sample_amount = 200
da = np.zeros((10, 10))


# Data Feed
# 784 (reshape=True) | 28*28 (reshape=False)
image_reshape = False
data_dir = "/home/ziyi/code/data/"

for i in range(10):
    mnist0 = dataset.read_data_sets(data_dir, target_class=i, one_hot=False,
                                    reshape=image_reshape, sample_vol=sample_amount)
    for j in range(10):
        if i==j : continue
      # print(mnist1.train.images.shape, mnist1.train.num_examples)

        mnist1 = dataset.read_data_sets(data_dir, target_class=j, one_hot=False,
                                        reshape=image_reshape, sample_vol=sample_amount*10)

        train_images = np.concatenate([mnist0.train.images, mnist1.train.images], 0)
        train_labels = np.concatenate([np.ones(mnist0.train.num_examples), -np.ones(mnist1.train.num_examples)])
        test_num = mnist0.test.num_examples
        test_images = np.concatenate([mnist0.test.images, mnist1.test.images], 0)
        test_labels = np.concatenate([np.ones(test_num), -np.ones(mnist1.test.num_examples)])
        clf0 = svm.SVC(kernel='linear')
        clf0.fit(train_images.reshape([-1, 28*28]), train_labels)
        prediction = clf0.predict(test_images.reshape([-1, 28*28]))
        acc = np.mean(prediction==test_labels)*1.
        print(acc)
        da[i,j] = acc

print(da)
