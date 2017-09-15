#!/bin/env python3

'''
This is a simple MLP script to test out the TensorBoard graph properties,
character constants, and weight matrix visualizers.
'''

import numpy as np
import tensorflow as tf
import pandas as pd

# grab data
df = pd.read_csv('../dat/mushroom_one_hot.csv')
dat = df.as_matrix()
df = pd.read_csv('../dat/mushroom_labels.csv')
lbl = df.as_matrix()

train_dat = dat[:8000,:]
train_lbl = lbl[:8000,:]
test_dat  = dat[8001:,:]
test_lbl  = lbl[8001:,:]

print(train_lbl)

# placeholders for I/O
inp = tf.placeholder('float', [None, 117], name='inp')
otp = tf.placeholder('float', [None, 1], name='otp')

# build weight matrices
w1 = tf.Variable(tf.random_normal([117,1000], stddev=0.01), name='w1')
w2 = tf.Variable(tf.random_normal([1000,500], stddev=0.01), name='w2')
w3 = tf.Variable(tf.random_normal([500,50], stddev=0.01), name='w3')
w4 = tf.Variable(tf.random_normal([50,1], stddev=0.01), name='w4')

# weight matrix histogram summaries
tf.summary.histogram('w1_summ',w1)
tf.summary.histogram('w2_summ',w2)
tf.summary.histogram('w3_summ',w3)
tf.summary.histogram('w4_summ',w4)

# build model
def model(inp, w1, w2, w3, w4):
	with tf.name_scope('layer1'):
		l1 = tf.nn.relu(tf.matmul(inp, w1))
	with tf.name_scope('layer2'):
		l2 = tf.nn.relu(tf.matmul(l1, w2))
	with tf.name_scope('layer3'):
		l3 = tf.nn.relu(tf.matmul(l2, w3))
	with tf.name_scope('layer4'):
		return tf.matmul(l3, w4)

mod = model(inp, w1, w2, w3, w4)

# build loss function
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mod, labels=otp))
	train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
	tf.summary.scalar('loss', loss)

# validate
with tf.name_scope('validation'):
	correct = tf.equal(tf.argmax(otp, 1), tf.argmax(mod, 1))
	val_op  = tf.reduce_mean(tf.cast(correct, 'float'))
	tf.summary.scalar('accuracy', val_op)

with tf.Session() as sess:
	# write out log
	writer = tf.summary.FileWriter('./logs/mlp', sess.graph)
	merged = tf.summary.merge_all()

	# init
	tf.global_variables_initializer().run()

	# train
	for i in range(10):
		sess.run(train_op, feed_dict={inp: train_dat, otp: train_lbl})

		summary, acc = sess.run([merged, val_op],
			feed_dict={inp: test_dat, otp: test_lbl})
		writer.add_summary(summary, i)

		print(i, acc)
		#print(tf.Tensor.eval(w1))
