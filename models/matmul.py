#!/bin/env python3

import numpy as np
import tensorflow as tf

d1 = np.random.rand(3,3)
d2 = np.random.rand(3,3)

one = tf.placeholder('float', [3,3], name='one')
two = tf.placeholder('float', [3,3], name='two')

with tf.name_scope('multiply'):
	mul_op = tf.matmul(one, two, name='mult')	
	tf.summary.scalar('thing', tf.reduce_max(mul_op))

tf.summary.histogram('d1_summ', d1)

img = np.reshape(d1, [1,3,3,1])
print(img)
tf.summary.image('d1_image', img)

with tf.Session() as sess:
	writer = tf.summary.FileWriter('./logs/matmul', sess.graph)
	merged = tf.summary.merge_all()

	tf.global_variables_initializer().run()

	for i in range(10):
		d2 = np.random.rand(3,3)

		summary, mul = sess.run([merged, mul_op], feed_dict={one: d1, two: d2})
		writer.add_summary(summary, i)

		#print(sess.run(tf.reduce_max(mul)), '\n', mul)
