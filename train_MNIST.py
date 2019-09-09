import os
import sys
import getopt

import numpy as np
import tensorflow as tf

import MNIST_FC
from read_MNIST import MNIST_IMG_X
from read_MNIST import MNIST_IMG_Y
from read_MNIST import load_mnist
from read_MNIST import BuildPipeline


############### global configs ################
layer_sizes = [512, 1024, 1024]
lr0 = 1e-2
lr = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
train_batch_size = 32
test_batch_size = 32
num_epochs = 50

dataset_path = 'MNIST'
# tensorboard summary will be saved as summary_path/summary_name
summary_path = './tensorboard/'
summary_name = 'summary-default'
# current optimal model checkpoint will be saved under model_path
model_path = './ckpts/'

# set global step counter
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

############### the input pipeline part ################
# pre-load numpy data
np_train_data, np_train_label = load_mnist(dataset_path, 'train')
np_test_data, np_test_label = load_mnist(dataset_path, 'test')

with tf.name_scope('input-pipeline'):
	# placeholder for feeding numpy data to dataset
	np_images_plh = tf.placeholder(dtype=tf.uint8, shape=(None, MNIST_IMG_X*MNIST_IMG_Y))
	np_labels_plh = tf.placeholder(dtype=tf.float32, shape=(None,))

	# construct datasets from numpy data
	train_dataset = BuildPipeline({'label': np_labels_plh, 'image': np_images_plh}, train_batch_size, 1)
	test_dataset = BuildPipeline({'label': np_labels_plh, 'image': np_images_plh}, test_batch_size, 1)

	# define iterators of datasets
	train_iterator = train_dataset.make_initializable_iterator()
	test_iterator = test_dataset.make_initializable_iterator()

	train_dataset_handle = train_iterator.string_handle()
	test_dataset_handle = test_iterator.string_handle()

	# define shared iterator for redirecion
	iterator_handle = tf.placeholder(dtype=tf.string, shape=None)
	iterator = tf.data.Iterator.from_string_handle(iterator_handle, train_iterator.output_types)

	images, labels = iterator.get_next()

############### build the FC pipeline ################
X = tf.placeholder(dtype=tf.uint8, shape=(None, MNIST_IMG_X*MNIST_IMG_Y))
Y = tf.placeholder(dtype=tf.uint8, shape=(None,))
logits_before_softmax = MNIST_FC.BuildFC_MNIST(X, layer_sizes)

loss = MNIST_FC.ComputeLoss(Y, logits_before_softmax)

# estimate the prediction accuracy
y = tf.argmax(tf.nn.softmax(logits_before_softmax), -1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, tf.cast(y, dtype=tf.uint8)), dtype=tf.uint8))

# optimize the loss
train_op = tf.train.AdamOptimizer(learning_rate=lr,
								  beta1=0.9,
								  beta2=0.999,
								  epsilon=1e-08).minimize(loss, global_step=global_step)

# define the training process
def train(cur_lr, sess, summary_writer, summary_op):
	'''
	input:
		cur_lr : learning rate for current epoch (scalar)
		sess : tf session to run the training process
		summary_writer : summary writer
		summary_op : summary to write in training process
	'''
	# get iterator handles
	train_dataset_handle_ = sess.run(train_dataset_handle)
	# re-initialize the iterator (because the dataset only repeat one epoch)
	sess.run(train_iterator.initializer, feed_dict={np_images_plh: np_train_data,
													np_labels_plh: np_train_label})
	# training loop
	current_batch = 0
	while True:
		try:
			# read batch of data from training dataset
			train_img_, train_label_ = sess.run([images,labels], feed_dict={np_images_plh: np_train_data,
																   np_labels_plh: np_train_label,
																   iterator_handle: train_dataset_handle_})
			# feed this batch to FC
			_, loss_, accuracy_, global_step_, summary_buff_ = \
				sess.run([train_op, loss, accuracy, global_step, summary_op],
						feed_dict={X : train_img_,
								   Y : train_label_,
								   lr: cur_lr})
			current_batch += 1
			# print indication info
			if current_batch % 100 == 0:
				print('\tbatch number = %d, loss = %.2f, acc = %.2f%%' % (current_batch, loss_, 100*accuracy_))
				# write training summary
				summary_writer.add_summary(summary=summary_buff_, global_step=global_step_)
		except tf.errors.OutOfRangeError:
			break
	# over

# build the test process
def test(sess, summary_writer):
	'''
	input :
		sess : tf session to run the validation
		summary_writer : summary writer
	'''
	# get iterator handles
	test_dataset_handle_ = sess.run(test_dataset_handle)
	# re-initialize the iterator (because the dataset only repeat one epoch)
	sess.run(test_iterator.initializer, feed_dict={np_images_plh: np_test_data, 
												   np_labels_plh: np_test_label})
	# validation loop
	correctness = 0
	loss_val = 0
	test_dataset_size = 0
	while True:
		try:
			# read batch of data from test dataset
			test_img_, test_label_ = sess.run([images,labels], feed_dict={np_images_plh: np_test_data,
																 np_labels_plh: np_test_label,
																 iterator_handle: test_dataset_handle_})
			cur_batch_size = test_img_.shape[0]
			test_dataset_size += cur_batch_size
			# test on single batch
			batch_accuracy_, batch_loss_, global_step_ = \
						sess.run([accuracy, loss, global_step], feed_dict={X : test_img_, 
																		   Y : test_label_})

			correctness += np.asscalar(batch_accuracy_*cur_batch_size*IMG_X*IMG_Y)
			loss_val += np.asscalar(loss_*cur_batch_size)
		except tf.errors.OutOfRangeError:
			break
	# compute accuracy and loss after a whole epoch
	current_acc = correctness/test_dataset_size/MNIST_IMG_X/MNIST_IMG_Y
	loss_val /= test_dataset_size
	# print and summary
	msg = 'test accuracy = %.2f%%, loss = %.2f' % (current_acc*100, loss_val)
	test_acc_summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy',simple_value=current_acc)])
	test_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=loss_val)])
	# write summary
	summary_writer.add_summary(summary=test_acc_summary, global_step=global_step_)
	summary_writer.add_summary(summary=test_loss_summary, global_step=global_step_)
	
	# print message
	print(msg)
	# over
	return current_acc


# simple function to adjust learning rate between epochs
def update_learning_rate(cur_epoch):
	'''
	input:
		epoch : current No. of epoch
	output:
		cur_lr : learning rate for current epoch
	'''
	cur_lr = lr0
	if cur_epoch > 10:
		cur_lr = lr0/10
	if cur_epoch >20:
		cur_lr = lr0/100
	if cur_epoch >30:
		cur_lr = lr0/1000
	if cur_epoch >40:
		cur_lr = lr0/2000
	# over
	return cur_lr


###################### main entrance ######################
if __name__ == "__main__":
	# set tensorboard summary path
	try:
		options, args = getopt.getopt(sys.argv[1:], '', ['logdir='])
	except getopt.GetoptError:
		print('invalid arguments!')
		sys.exit(-1)
	for option, value in options:
		if option == '--logdir':
			summary_name = value

	# train and test the model
	cur_lr = lr0
	best_acc = 0
	with tf.Session() as sess:
		# initialize variables
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		# initialize IO
		# build the tensorboard summary
		summary_writer = tf.summary.FileWriter(summary_path+summary_name)
		train_summary_op = tf.summary.merge_all()

		# train in epochs
		for cur_epoch in range(1, num_epochs+1):
			# print epoch title
			print('Current epoch No.%d, learning rate = %.2e' % (cur_epoch, cur_lr))
			# train
			train(cur_lr, sess, summary_writer, train_summary_op)
			# validate
			cur_acc = test(sess, summary_writer)
			# update learning rate if necessary
			cur_lr = update_learning_rate(cur_epoch)

			if cur_acc > best_acc:
				# print message
				print('model improved, save the ckpt.')
				# update best loss
				best_acc = cur_acc
			else:
				# print message
				print('model not improved.')
	# finished
	print('++++++++++++++++++++++++++++++++++++++++')
	print('best accuracy = %.2f%%.'%(best_acc*100))