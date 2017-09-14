#!/bin/env python3

import numpy as np
import tensorflow as tf

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

print(train_labels.shape) # (55000,)
print(train_data.shape) # (55000, 784)

# placeholders
data = tf.placeholder('float', [None, 784], name='data')
labels = tf.placeholder('float', [None, 1], name='data')

def model(data):
	with tf.name_scope('input'):
		layer1 = tf.reshape(data, [-1, 28, 28, 1])
	with tf.name_scope('conv1'):
		conv1 = tf.layers.conv2d(inputs=layer1, filters=32, 
			kernel_size=[5,5], padding='same', activation=tf.nn.relu)
	with tf.name_scope('pool1'):
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],
			strides=2)
	with tf.name_scope('conv2'):
		conv2 = tf.layers.conv2d(inputs=pool1, filters=64, 
			kernel_size=[5,5], padding='same', activation=tf.nn.relu)
	with tf.name_scope('pool2'):
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2],
			strides=2)
	with tf.name_scope('flatten'):
		flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
	with tf.name_scope('dense'):
		dense = tf.layers.dense(inputs=flat, 
			units=1024, activation=tf.nn.relu)
	with tf.name_scope('output'):
		output = tf.layers.dense(inputs=dense, units=10)

mod = model(data)

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
	
