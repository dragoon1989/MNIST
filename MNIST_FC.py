''' build a FC nn on MNISt '''
import tensorflow as tf


# 
def BuildFC_MNIST(X, layer_sizes):
	''' input:	X --- input MNIST data as batch of 1D vector (dtype=tf.uint8)
				layer_size --- input 1D list, containing each hidden FC layer's size
		output:	logits_before_softmax --- predicted logits before softmax (shape=(B,10), dtype=tf.float32)
	'''
	# preprocess input
	X = tf.to_float(X)
	
	X = tf.reshape(X, shape=(-1,28*28))
	# build FC layers
	layer_id = 0
	in_size = 28*28
	with tf.variable_scope('FC-layer'):
		# initializer
		w_init = tf.initializers.he_normal()
		b_init = tf.initializers.constant(0.0)
		# hidden layers
		feature = X
		for layer_size in layer_sizes:
			# get layer weights
			w = tf.get_variable(name='fc-'+str(layer_id)+'-w',
								shape=(in_size, layer_size),
								dtype=tf.float32, 
								initializer=w_init,
								trainable=True)
			# get layer biases
			b = tf.get_variable(name='fc-'+str(layer_id)+'-b',
								shape=(layer_size,),
								dtype=tf.float32,
								initializer=b_init,
								trainable=True)
			# apply the layer
			feature = tf.matmul(feature,w) + b
			# activation
			feature = tf.nn.relu(feature)
			# update 
			in_size = layer_size
		# output layer
		w_out = tf.get_variable(name='output-w',
								shape=(in_size, 10),
								dtype=tf.float32,
								initializer=w_init,
								trainable=True)
		b_out = tf.get_variable(name='output-b',
								shape=(10,),
								dtype=tf.float32,
								initializer=b_init,
								trainable=True)
		logits_before_softmax = tf.matmul(feature,w_out) + b_out
	# over
	return logits_before_softmax

#
def ComputeLoss(labels, logits_before_softmax):
	''' input:	labels --- input batch of labels (shape=(B,), dtype=tf.uint8)
				logits_before_softmax --- input logits before softmax (shape=(B,10),dtype=tf.float32)
		output:	loss --- scalar averaged loss over batch (dtype=tf.float32)
	'''
	# the sparse_softmax_cross_entropy_with_logits only accept labels with dtype=tf.int32 or tf.int64
	labels = tf.cast(labels, dtype=tf.int32)
	batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
							logits=logits_before_softmax, name='batch-loss')
	# average over the batch
	loss = tf.reduce_mean(batch_loss)
	# over
	return loss

		