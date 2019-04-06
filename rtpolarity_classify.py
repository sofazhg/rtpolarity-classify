#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/5 21:53
# @Author  : ZHANG Shaohua
# @Contact : sofazhg@outlook.com
# @File    : rtpolarity_classify.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from data_helpers import load_data_and_labels

import re
import numpy as np
import pandas as pd
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


'''
def word_to_vec(x_text):
    vocab = ''
    for text in x_text:
        vocab += text.split()
    model = Word2Vec(vocab, sg=0, size=192, min_count=5, workers=4)
    model.save(r'./model/rt-polaritydata.word2vec')
'''

positive_data_file = open(r'./data/rt-polaritydata/rt-polarity.pos', "r", encoding='utf-8')
negative_data_file = open(r'./data/rt-polaritydata/rt-polarity.neg', "r", encoding='utf-8')
x_text, y = load_data_and_labels(positive_data_file, negative_data_file)
X_train, X_test, y_train, y_test = train_test_split(x_text, y, test_size=0.1, random_state=42)
num_words = len(x_text)
num_labels = len(y)
print("training entries: {}, labels: {}".format(num_words, num_labels))
print(X_train[0])

# 分词，构建单词-id词典
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tokenizer.fit_on_texts(x_text)
vocab = tokenizer.word_index
print('vocab size:' + str(len(vocab)) + '\n')

# 将每个词用词典中的数值代替
X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)

# One-hot
x_train = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
x_test = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')

# 序列模式
x_train = pad_sequences(X_train_word_ids, maxlen=256)
x_test = pad_sequences(X_test_word_ids, maxlen=256)

# 构建模型
model = keras.Sequential()
model.add(keras.layers.Embedding(len(vocab) + 1, 16))
model.add(keras.layers.GlobalAveragePooling1D())  # 对序列维度求平均，为每个示例返回固定长度的输出向量
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# 显示模型的概况
model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:2000]
partial_x_train = x_train[2000:]

y_val = y_train[:2000]
partial_y_train = y_train[2000:]

# 训练
history = model.fit(partial_x_train, partial_y_train,
                    epochs=40,
                    batch_size=256,
                    validation_data=(x_val, y_val),
                    verbose=1)

model.save('./model/rtpolarity_classify.h5')
results = model.evaluate(x_test, y_test)
print(results)
