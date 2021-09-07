#!/usr/bin/env python3

# load needed modules

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random as rn


np.random.seed(123)
rn.seed(123)
tf.random.set_seed(123)

# load feature data
X = np.load('data/x.npy')
y = np.load('data/y.npy')

# reshape x untuk lstm
X = X.reshape((X.shape[0], 1, X.shape[1]))

# if labels are not in integer, convert it, otherwise comment it
y = y.astype(int)

# split into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                             patience=200,
                                             restore_best_weights=True)
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)


# function to define model
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(axis=-1,
              input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    # compile model: set loss, optimizer, metric
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


# create the model
model = create_model()
print(model.summary())

# train the model
hist = model.fit(x_train, y_train, epochs=200, shuffle=True, batch_size=16)
evaluate = model.evaluate(x_test, y_test, batch_size=16)
print(evaluate)

# make prediction for confusion_matrix
# import os
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# predict = model.predict(test_x, batch_size=16)
# emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# # predicted emotions from the test set
# y_pred = np.argmax(predict, 1)
# predicted_emo = []
# for i in range(0,test_y.shape[0]):
#     emo = emotions[y_pred[i]]
#     predicted_emo.append(emo)

# # get actual emotion
# actual_emo = []
# y_true = np.argmax(test_y, 1)
# for i in range(0,test_y.shape[0]):
#     emo = emotions[y_true[i]]
#     actual_emo.append(emo)

# # generate the confusion matrix
# cm = confusion_matrix(actual_emo, predicted_emo)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# #index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# #columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# #cm_df = pd.DataFrame(cm, index, columns)
# #plt.figure(figsize=(10, 6))
# #sns.heatmap(cm_df, annot=True)
# #plt.savefig('speech_librosa_hfs.svg')
# print("UAR: ", cm.trace()/cm.shape[0])
