#!/usr/bin/python3

# Authors:	Peter Schoener, 4013996
#			Luana Vaduva, 3974913
# Honor Code: We pledge that this program represents our own work.

from enum import Enum
import os
import sys
import re

import numpy as np
import tensorflow as tf
import gensim
from sklearn import metrics

from bs4 import BeautifulSoup
# from guess_language import guess_language

from config import DefaultConfig
from model import Model, Phase
from numberer import Numberer

def preprocess(text):
	#convert to lower case
	text = text.lower()
	#convert www.* or https?://* to URL
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)
	#convert @username to AT_USER
	text = re.sub('@[^\s]+','AT_USER',text)
	#remove additional white spaces
	text = re.sub('[\s]+', ' ', text)
	#replace #word with word
	text = re.sub(r"#", r"", text) # should be equivalent for our purposes
	#trim
	text = text.strip('\'"')
	#get rid of HTML markup
	soup = BeautifulSoup(text, "html5lib")

	return soup.get_text()


def read_lexicon(filename):
	with open(filename, "r") as f:
		lex = {}
		
		for line in f:
			fields = line.split("\t")
			if len(fields) > 3 :
				lex[preprocess(fields[1])] = {fields[3].strip():1.0}
		return lex


def recode_lexicon(lexicon, words, labels, train=False):
	int_lex = []

	for (sentence, tags) in lexicon.items():
		int_sentence = []
		for word in sentence.split():
			int_sentence.append(words.number(word, train))

		int_tags = {}
		for (tag, p) in tags.items():
			int_tags[labels.number(tag, train)] = p

		int_lex.append((int_sentence, int_tags))

	return int_lex


def generate_instances(
		data,
		max_label,
		max_timesteps,
		batch_size=128):
	n_batches = len(data) // batch_size

	# We are discarding the last batch for now, for simplicity.
	labels = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_label.max_number()),
		dtype=np.float32)
	lengths = np.zeros(
		shape=(
			n_batches,
			batch_size),
		dtype=np.int32)
	sentences = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_timesteps),
		dtype=np.int32)

	for batch in range(n_batches):
		for idx in range(batch_size):
			(sentence, l) = data[(batch * batch_size) + idx]

			# Add label distribution
			for label, prob in l.items():
				labels[batch, idx, label] = prob

			# Sequence
			timesteps = min(max_timesteps, len(sentence))

			# Sequence length (time steps)
			lengths[batch, idx] = timesteps

			# Word characters
			sentences[batch, idx, :timesteps] = sentence[:timesteps]

	return (sentences, lengths, labels)


def train_model(config, train_batches, validation_batches, numberer):
	train_batches, train_lens, train_labels = train_batches
	validation_batches, validation_lens, validation_labels = validation_batches

	n_chars = max(np.amax(validation_batches), np.amax(train_batches)) + 1

	with tf.Session() as sess:
		with tf.variable_scope("model", reuse=False):
			train_model = Model(
				config,
				train_batches,
				train_lens,
				train_labels,
				n_chars,
				numberer,
				phase=Phase.Train)

		with tf.variable_scope("model", reuse=True):
			validation_model = Model(
				config,
				validation_batches,
				validation_lens,
				validation_labels,
				n_chars,
				numberer,
				phase=Phase.Validation)

		sess.run(tf.global_variables_initializer())

		print()
		print("       | train   |                   validation                    |")
		print(" epoch | loss    | loss    | acc.    | prec.   | recall  | F1      |")
		print("-------+---------+---------+---------+---------+---------+---------+")
		for epoch in range(config.n_epochs):
			train_loss = 0.0
			validation_loss = 0.0
			accuracy = 0.0
			precision = 0.0
			recall = 0.0
			f1 = 0.0

			# Train on all batches.
			for batch in range(train_batches.shape[0]):
				loss, _ = sess.run([train_model.loss, train_model.train_op], {
					train_model.x: train_batches[batch], train_model.lens: train_lens[batch], train_model.y: train_labels[batch]})
				train_loss += loss

			# validation on all batches.
			for batch in range(validation_batches.shape[0]):
				loss, acc, hpl = sess.run([validation_model.loss, validation_model.accuracy, validation_model.hp_labels], {
					validation_model.x: validation_batches[batch], validation_model.lens: validation_lens[batch], validation_model.y: validation_labels[batch]})
				validation_loss += loss
				accuracy += acc
				precision += metrics.precision_score(np.argmax(np.array(validation_labels[batch]).astype(np.int32), axis = 1), hpl, average = "macro")
				recall += metrics.recall_score(np.argmax(np.array(validation_labels[batch]).astype(np.int32), axis = 1), hpl, average = "macro")
				f1 += metrics.f1_score(np.argmax(np.array(validation_labels[batch]).astype(np.int32), axis = 1), hpl, average = "macro")

			train_loss /= train_batches.shape[0]
			validation_loss /= validation_batches.shape[0]
			accuracy /= validation_batches.shape[0]
			precision /= validation_batches.shape[0]
			recall /= validation_batches.shape[0]
			f1 /= validation_batches.shape[0]

			print(" % 3d   | % 4.2f | % 4.2f | % 2.2f%% | % 2.2f%% | % 2.2f%% | % 2.2f%% |" % (epoch, train_loss, validation_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))


if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.stderr.write("Usage: %s TRAIN_SET DEV_SET\n" % sys.argv[0])
		sys.exit(1)

	config = DefaultConfig()

	# Read training, validation, and embedding data.
	train_lexicon = read_lexicon(sys.argv[1])
	validation_lexicon = read_lexicon(sys.argv[2])

	# Convert word characters and part-of-speech labels to numeral representation.
	words = Numberer()
	labels = Numberer()
	train_lexicon = recode_lexicon(train_lexicon, words, labels, train=True)
	validation_lexicon = recode_lexicon(validation_lexicon, words, labels)

	# Generate batches
	train_batches = generate_instances(
		train_lexicon,
		labels,
		config.max_timesteps,
		batch_size=config.batch_size)
	validation_batches = generate_instances(
		validation_lexicon,
		labels,
		config.max_timesteps,
		batch_size=config.batch_size)

	# Train the model
	train_model(config, train_batches, validation_batches, words)
