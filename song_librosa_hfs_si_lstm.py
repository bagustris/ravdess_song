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
X_train = np.load('data/x_train.npy')
X_test = np.load('data/x_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

# reshape x untuk lstm
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             restore_best_weights=True)
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)


# function to define model
def model_lstm():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(axis=-1,
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))

    # compile model: set loss, optimizer, metric
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


# create the model
model = model_lstm()
print(model.summary())

# train the model
hist = model.fit(X_train, 
                 y_train, 
                 epochs=100, 
                 shuffle=True,
                 callbacks=earlystop,
                 validation_split=0.1,
                 batch_size=16)
evaluate = model.evaluate(X_test, y_test, batch_size=16)
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
