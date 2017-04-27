from __future__ import print_function
import numpy as np
import pandas as pd
import itertools
import glob

from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Embedding, GlobalMaxPooling1D, Dropout, Convolution1D,\
    Input, LSTM, merge, concatenate, SpatialDropout1D, Conv2D, Conv1D
from keras.layers.advanced_activations import PReLU
import sklearn.cross_validation
# fix random seed for reproducibility
np.random.seed(1337)


# find maxlen function
def longest(list1):
    longest_list = max(len(elem) for elem in list1)
    return longest_list


print("Loading data....")
data_files = glob.glob('data/*.txt')
d = []
for i in data_files:
    print('Loading....', i)
    line = np.loadtxt(i)
    d.append(line)
    # print("d......", d)

# find maxlen in data
maxlen = longest(d)
print('Max lenght is :', maxlen)
print('Data Sequences :', len(d))

# -------Padding------
print("Padding sequences...")  # adding zeros
# X_train1 = sequence.pad_sequences(d[:int(len(d)*0.8)], maxlen=maxlen)
# X_test1 = sequence.pad_sequences(d[int(len(d)*0.8):], maxlen=maxlen)
d_pad = sequence.pad_sequences(d, maxlen=200)
# -------Reverse------
d_reversed = d_pad[:, :, ::-1]  
# -------Merge------
data = np.append(d_pad, d_reversed, axis=0)  
print('Data Sequences :', len(data))


# -------Label Information------
# define label 1 class value for x->y causal direction
labels_1 = np.ones(len(d))
# define label 1 class value for x->y causal direction
labels_0 = np.zeros(len(d))
targets = np.append(labels_1, labels_0, axis=0)
print('Targets : ', len(targets))


# -------Cross Validation-------
ss = sklearn.cross_validation.ShuffleSplit(len(data),
                                           n_iter=10,
                                           test_size=.25,
                                           random_state=1234)
fold = 0
for train_index, test_index in ss:
    fold += 1
    print("Fold", fold, "--------------------------------")

    # get the data and label for the training data
    X_train = np.array([data[x] for x in train_index])
    y_train = np.array([targets[x] for x in train_index])

    # label and data for the test data
    X_test = np.array([data[x] for x in test_index])
    y_test = np.array([targets[x] for x in test_index])

    # -------Model------

    batch_size = 10  # denotes the subset size of training sample

    print('Build model...')

    inputs = Input(shape=(X_train.shape[1:]), )  # shape input
    x = inputs
    x = Convolution1D(64, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(64)(x)
    x = Activation("relu")(x)
    x = Dense(1)(x)  # project onto a single unit output layer, and squash it with a sigmoid:
    x = Activation('sigmoid')(x)
    predictions = Activation("sigmoid")(x)

    model = Model(input=inputs, output=predictions)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # -------Training------
	print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=15,
              validation_data=(X_test, y_test))
    # -------Testing------
	score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print(' ')
    print('Test score:', score)
    print('Test accuracy:', acc)





