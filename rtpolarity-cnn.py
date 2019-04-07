#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/7 14:14
# @Author  : ZHANG Shaohua
# @Contact : sofazhg@outlook.com
# @File    : rtpolarity-cnn.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from data_helpers import load_data_and_labels
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, BatchNormalization, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

positive_data_file = open(r'./data/rt-polaritydata/rt-polarity.pos', "r", encoding='utf-8')
negative_data_file = open(r'./data/rt-polaritydata/rt-polarity.neg', "r", encoding='utf-8')
x_text, y = load_data_and_labels(positive_data_file, negative_data_file)
X_train, X_test, y_train, y_test = train_test_split(x_text, y, test_size=0.2, random_state=42)
num_entries = len(x_text)
num_labels = len(y)
print("training entries: {}, labels: {}".format(num_entries, num_labels))
print(X_train[0])

# 分词，构建单词-id词典
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tokenizer.fit_on_texts(x_text)
vocab = tokenizer.word_index
print('vocab size:' + str(len(vocab)) + '\n')

# 将每个词用词典中的数值代替
x_train_word_ids = tokenizer.texts_to_sequences(X_train)
x_test_word_ids = tokenizer.texts_to_sequences(X_test)

# One-hot
x_train = tokenizer.sequences_to_matrix(x_train_word_ids, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test_word_ids, mode='binary')

# 序列模式
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=64)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=64)

# CNN model
model = Sequential()
model.add(Embedding(len(vocab) + 1, 256, input_length=64))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(256, 3, padding='same'))
model.add(MaxPool1D(3, 3, padding='same'))
model.add(Convolution1D(128, 3, padding='same'))
model.add(MaxPool1D(3, 3, padding='same'))
model.add(Convolution1D(64, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          validation_split=0.25,
          epochs=12,
          batch_size=32,
          verbose=1)

model.save('./model/rtpolarity-cnn.h5')
scores = model.evaluate(x_test_padded_seqs, y_test)
print(scores)
