#!/bin/env python3

import numpy as np
import tensorflow as tf

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# TODO: this is the part***************************************************************
#np.expand_dims(train_labels, axis=0)
np.transpose(train_labels)

print(train_labels.shape) # (55000,)
print(train_labels)
print(train_data.shape) # (55000, 784)

# placeholders
data = tf.placeholder('float', [None, 784], name='data')
labels = tf.placeholder('float', [None, 1], name='labels')

# weight matrices and bias vectors
w1 = tf.Variable(tf.random_normal([5,5,1,32], stddev=1), name='w1')
w2 = tf.Variable(tf.random_normal([5,5,32,64], stddev=1), name='w2')
w3 = tf.Variable(tf.random_normal([7*7*64, 1024], stddev=1), name='w3')
w4 = tf.Variable(tf.random_normal([1024, 10], stddev=1), name='w4')

b1 = tf.Variable(tf.random_normal([32], stddev=1), name='b1')
b2 = tf.Variable(tf.random_normal([64], stddev=1), name='b2')

# weight matrix histogram summaries
tf.summary.histogram('w1_summ', w1)
tf.summary.histogram('w2_summ', w2)
tf.summary.histogram('w3_summ', w3)
tf.summary.histogram('w4_summ', w4)

tf.summary.histogram('b1_summ', b1)
tf.summary.histogram('b2_summ', b2)

# build the model
def model(data,w1,w2,w3,w4,b1,b2):
	with tf.name_scope('input'):
		layer1 = tf.reshape(data, [-1, 28, 28, 1])
	with tf.name_scope('conv_1'):
		conv1 = tf.nn.conv2d(layer1, filter=w1, strides=[1,1,1,1], padding='SAME')
		conv1b = tf.nn.relu(conv1 + b1)
		pool1 = tf.nn.max_pool(conv1b, ksize=[2,2,1,1], strides=[1,1,1,1], padding='SAME')
	with tf.name_scope('conv_2'):
		conv2 = tf.nn.conv2d(pool1, filter=w2, strides=[1,1,1,1], padding='SAME')
		conv2b = tf.nn.relu(conv2 + b2)
		pool2 = tf.nn.max_pool(conv2b, ksize=[2,2,1,1], strides=[1,1,1,1], padding='SAME')
	with tf.name_scope('flatten'):
		flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
	with tf.name_scope('dense'):
		dense1 = tf.nn.relu(tf.matmul(flat, w3))
	with tf.name_scope('output'):
		return tf.matmul(dense1, w4)

mod = model(data,w1,w2,w3,w4,b1,b2)

# build loss function
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mod, labels=labels))
	train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
	tf.summary.scalar('loss', loss)

# validate
with tf.name_scope('validation'):
	correct = tf.equal(tf.argmax(labels, 1), tf.argmax(mod, 1))
	val_op  = tf.reduce_mean(tf.cast(correct, 'float'))
	tf.summary.scalar('accuracy', val_op)

# run it
with tf.Session() as sess:
	# write out log
	writer = tf.summary.FileWriter('./logs/cnn', sess.graph)
	merged = tf.summary.merge_all()

	# init
	tf.global_variables_initializer().run()

	# train
	for i in range(2):
		sess.run(train_op, feed_dict={data: train_data, labels: train_labels})

		summary, acc = sess.run([merged, val_op],
			feed_dict={data: test_data, labels: test_labels})
		writer.add_summary(summary, i)

		print(i, acc)
