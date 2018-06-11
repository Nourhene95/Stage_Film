import numpy as np
np.random.seed(1000)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  #ou '0' selon que tu veux le premier ou second GPU d'hydra
import tensorflow as tf
sc = tf.ConfigProto()
sc.gpu_options.allow_growth = True
s = tf.Session(config=sc)
from keras import backend as K
K.set_session(s)


import tensorflow as tf
import timeit
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt',encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
filename = 'trad_1.txt'
doc = load_doc(filename)
# split into english-french pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-french.pkl')
# spot check
for i in range(100):
	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))

from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
 
# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
 
# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)
 
# load dataset
raw_dataset = load_clean_sentences('english-french.pkl')

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb')) 

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)
from keras.preprocessing.text import Tokenizer

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential 
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from pickle import load
from numpy import array
from numpy import argmax
from keras.layers import RepeatVector
from keras.layers import GRU
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
  model = Sequential()
  model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
  model.add(LSTM(n_units))
  model.add(RepeatVector(tar_timesteps))
  model.add(GRU(n_units, return_sequences=True))
  
  model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
  return model


# prepare english tokenizer
eng_tokenizer = create_tokenizer(raw_dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(raw_dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare french tokenizer
fre_tokenizer = create_tokenizer(raw_dataset[:, 1])
fre_vocab_size = len(fre_tokenizer.word_index) + 1
fre_length = max_length(raw_dataset[:, 1])
print('French Vocabulary Size: %d' % fre_vocab_size)
print('French Max Length: %d' % (fre_length))

# define model
model = define_model(eng_vocab_size, fre_vocab_size, eng_length, fre_length, 350)
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer='adam', loss='categorical_crossentropy')

# summarize defined model
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'model.h5'
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)
# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
  actual, predicted = list(), list()
  for i, source in enumerate(sources):
    # translate encoded source text
    source = source.reshape((1, source.shape[0]))
    translation = predict_sequence(model, fre_tokenizer, source)
    raw_target, raw_src = raw_dataset[i]
    if i < 10:
      print('src=[%s], target=[%s], predicted=[%s]' % (raw_target,raw_src, translation))
    actual.append(raw_src.split())
    predicted.append(translation.split())
  # calculate BLEU score
#  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
#  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
#  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
#  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

for i in range (5):
        # reduce dataset size
        n_debut=i*10000
        n_sentences = (i+1)*10000
        dataset = raw_dataset[n_debut:n_sentences, :]
        # random shuffle
        shuffle(dataset)
        # split into train/val        
        train, val = dataset[:9000], dataset[9000:]
        # save
        save_clean_data(dataset, 'english-french-both.pkl')
        save_clean_data(train, 'english-french-train.pkl')
        save_clean_data(val, 'english-french-val.pkl')
        # load datasets
        dataset = load_clean_sentences('english-french-both.pkl')
        train = load_clean_sentences('english-french-train.pkl')
        test = load_clean_sentences('english-french-val.pkl')
        # prepare english tokenizer
        eng_tokenizer = create_tokenizer(dataset[:, 0])
        #eng_vocab_size = len(eng_tokenizer.word_index) + 1
        #eng_length = max_length(dataset[:, 0])
        #print('English Vocabulary Size: %d' % eng_vocab_size)
        #print('English Max Length: %d' % (eng_length))
        # prepare french tokenizer
        fre_tokenizer = create_tokenizer(dataset[:, 1])
        #fre_vocab_size = len(fre_tokenizer.word_index) + 1
        #fre_length = max_length(dataset[:, 1])
        #print('French Vocabulary Size: %d' % fre_vocab_size)
        #print('French Max Length: %d' % (fre_length))

        # prepare training data
        trainY = encode_sequences(fre_tokenizer, fre_length, train[:, 1])
        trainX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
        trainY = encode_output(trainY, fre_vocab_size)
        # prepare validation data
        valY = encode_sequences(fre_tokenizer, fre_length, test[:, 1])
        valX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
        valY = encode_output(valY, fre_vocab_size)
        model.fit(trainX, trainY, epochs=30, batch_size=32, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
        #model.fit(trainX, trainY, epochs=10, batch_size=64, validation_data=(valX, valY), callbacks=[learning_rate_reduction], verbose=2)
        #save model
        model.save(filename)
        # test on some training sequences
        evaluate_model(model, fre_tokenizer, trainX, train)
        # test on some test sequences
        evaluate_model(model, fre_tokenizer, valX, test)
        







