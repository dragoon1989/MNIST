import os
import struct
import gzip

import numpy as np
import tensorflow as tf


MNIST_IMG_X = 28
MNIST_IMG_Y = 28

# module to load images and labels
# we only have .gz package, so use gzip to read the binary file
def load_mnist(path, kind='train'):
	''' input:	path --- path to the MNIST .gz files
				kind --- 'train' or 'test'
		output: images --- images stored as batch of 1D vectors (shape=(N,28*28), dtype=np.uint8)
				labels --- labels sotred as batch of scalars (shape=(N,), dtype=np.uint8) '''
	labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
	images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)
	# read labels
	with gzip.open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II', lbpath.read(8))
		#labels = np.fromfile(lbpath, dtype=np.uint8)
		format = '>'+str(n)+'B'
		labels = struct.unpack(format, lbpath.read())
		# convert label to numpy uint8 array
		labels = np.array(labels).astype(np.uint8)
	# read images
	with gzip.open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
		#images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
		format = '>'+str(num*rows*cols)+'B'
		images = struct.unpack(format, imgpath.read())
		# reshape and cast type to uint8
		images = np.reshape(images, [num, rows*cols]).astype(np.uint8)
	# over
	return images, labels


# build a pipeline to convert MINIST data from numpy arrays to tf dataset
def BuildPipeline(raw_data,
				  batch_size,
				  num_epoch=1):
	''' input: raw_data	raw numpy MNIST data, recommend to use a python dict with 
					label: label data; image: image data
			   batch_size
			   num_epoch
		output: dataset		a dataset '''
	# slice the numpy data
	dataset = tf.data.Dataset.from_tensor_slices(raw_data)
	# repeat num_epoch 
	dataset = dataset.repeat(num_epoch)
	# shuffle
	dataset = dataset.shuffle(10*batch_size)
	# batch
	dataset = dataset.batch(batch_size)
	# over
	return dataset


