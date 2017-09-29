from enum import Enum

import tensorflow as tf
import numpy
from tensorflow.contrib import rnn

class Phase(Enum):
	Train = 0
	Validation = 1
	Predict = 2

class Model:
	def __init__(self, config, batch, lens_batch, label_batch, n_chars, numberer, phase = Phase.Predict):
		batch_size = batch.shape[1]
		input_size = batch.shape[2]
		label_size = label_batch.shape[2]
		
		# The integer-encoded words. input_size is the (maximum) number of
		# time steps.
		self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

		# This tensor provides the actual number of time steps for each
		# instance.
		self._lens = tf.placeholder(tf.int32, shape=[batch_size])

		# The label distribution.
		if phase != Phase.Predict:
			self._y = tf.placeholder(
				tf.float32, shape=[batch_size, label_size])

		# convert to embeddings
		embeddings = tf.get_variable("embeddings", shape = [n_chars, config.embedding_sz])
		input_layer = tf.nn.embedding_lookup(embeddings, self._x)

		# make a bunch of LSTM cells and link them
		# use rnn.DropoutWrapper instead of tf.nn.dropout because the layers are anonymous
		stacked_LSTM = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(config.LSTM_sz), output_keep_prob = config.dropout_ratio) for _ in range(config.LSTM_ct)])
				
		# run the whole thing
		_, hidden = tf.nn.dynamic_rnn(stacked_LSTM, input_layer, sequence_length = self._lens, dtype = tf.float32)
		w = tf.get_variable("W", shape=[hidden[-1].h.shape[1], label_size]) # if I understood the structure of MultiRNNCell correctly, hidden[-1] should be the final state
		b = tf.get_variable("b", shape=[1])
		logits = tf.matmul(hidden[-1].h, w) + b

		if phase == Phase.Train or Phase.Validation:
			losses = tf.nn.softmax_cross_entropy_with_logits(
				labels=self._y, logits=logits)
			self._loss = loss = tf.reduce_sum(losses)

		if phase == Phase.Train:
			start_lr = 0.005
			self._train_op = tf.train.AdamOptimizer(start_lr).minimize(losses)
			self._probs = probs = tf.nn.softmax(logits)

		if phase == Phase.Validation:
			# Highest probability labels of the gold data.
			gs_labels = tf.argmax(self._y, axis=1)

			# Predicted labels
			self._hp_labels = tf.argmax(logits, axis=1)

			correct = tf.equal(self._hp_labels, gs_labels)
			correct = tf.cast(correct, tf.float32)
			self._accuracy = tf.reduce_mean(correct)
			
			#self._hp_labels = hp_labels

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def hp_labels(self) :
		return self._hp_labels
	
	@property
	def lens(self):
		return self._lens

	@property
	def loss(self):
		return self._loss

	@property
	def probs(self):
		return self._probs

	@property
	def train_op(self):
		return self._train_op

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y
