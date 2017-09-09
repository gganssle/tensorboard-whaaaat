#!/bin/env python3

import numpy as np
import tensorflow as tf

d1 = np.random.rand(3,3)
d2 = np.random.rand(3,3)

one = tf.placeholder('float', [3,3], name='one')
two = tf.placeholder('float', [3,3], name='two')

with tf.name_scope('multipliy'):
	mul_op = tf.matmul(one, two, name='mult')
	tf.summary.scalar('thing', mul_op)

with tf.Session() as sess:
	writer = tf.summary.FileWriter('./logs/matmul', sess.graph)
	merged = tf.summary.merge_all()

	tf.global_variables_initializer().run()

	sess.run(mul_op, feed_dict={one: d1, two: d2})
	#summary = sess.run([merged, mul_op], feed_dict={one: d1, two: d2})
	writer.add_summary(summary)
