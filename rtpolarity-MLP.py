#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/7 14:23
# @Author  : ZHANG Shaohua
# @Contact : sofazhg@outlook.com
# @File    : rtpolarity-MLP.py
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


# 构建MLP模型
model = keras.Sequential()
model.add(keras.layers.Embedding(len(vocab) + 1, 16))
model.add(keras.layers.GlobalAveragePooling1D())  # 对序列维度求平均，为每个示例返回固定长度的输出向量
model.add(keras.layers.Dense(units=16, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=1, activation='sigmoid'))

# 显示模型的概况
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练
history = model.fit(x_train_padded_seqs, y_train,
                    validation_split=0.25,
                    epochs=40,
                    batch_size=512,
                    verbose=1)

model.save('./model/rtpolarity-mlp.h5')
scores = model.evaluate(x_test_padded_seqs, y_test)
print(scores)
