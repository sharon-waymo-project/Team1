#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import random
import glob


# In[2]:


epochs = 300
n_hidden = 128
OBSERVE_LENGTH = 10
PREDICT_LENGTH = 5
dim_input = 15
learning_rate = 0.0001
batch_size = 64

model_name = 'sslstm_epoch_{}_hidden_{}_observe_{}_predict_{}'.format(epochs, n_hidden, OBSERVE_LENGTH, PREDICT_LENGTH)


# In[3]:


TRAIN_FOLDERS = '/home/dataset/training_observe_{}_predict_{}/*/'.format(OBSERVE_LENGTH, PREDICT_LENGTH)
TEST_FOLDERS = '/home/dataset/validation_observe_{}_predict_{}/*/'.format(OBSERVE_LENGTH, PREDICT_LENGTH)

train_folders = glob.glob(TRAIN_FOLDERS)
val_ratio = 0.1

print('train folder num:', len(train_folders))


# In[4]:


test_folders = glob.glob(TEST_FOLDERS)
test_num = len(test_folders)
print('test folder num:', len(test_folders))


# In[5]:


train_X = []
train_y = []
for folder in train_folders:
    file_x = folder + 'X.npy'
    train_X.append(np.load(file_x))
    
    file_y = folder + 'y.npy'
    train_y.append(np.load(file_y))
    
train_X = np.array(train_X)
train_y = np.array(train_y)

test_X = []
test_y = []
for folder in test_folders:
    file_x = folder + 'X.npy'
    test_X.append(np.load(file_x))
    
    file_y = folder + 'y.npy'
    test_y.append(np.load(file_y))
    
test_X = np.array(test_X)
test_y = np.array(test_y)


print("train X: ", train_X.shape)
print("train y: ", train_y.shape)

print("test X: ", test_X.shape)
print("test y: ", test_y.shape)


# In[6]:


def build_model():
    opt = optimizers.RMSprop(lr=learning_rate)
    model = tf.keras.Sequential()
    #lstm encoder
    model.add(layers.GRU(n_hidden,
                  input_shape=(OBSERVE_LENGTH, dim_input),
                  return_sequences=False,
                  stateful=False,
                  dropout=0.2))
    model.add(layers.RepeatVector(PREDICT_LENGTH))
    #lstm decoder
    model.add(layers.GRU(n_hidden,
                  return_sequences=True,
                  stateful=False,
                  dropout=0.2))
    model.add(layers.TimeDistributed(layers.Dense(3), input_shape=(PREDICT_LENGTH, None)))
    model.add(layers.Activation('linear'))
    model.compile(loss='mse', optimizer=opt)
    
    print(model.summary())
    return model


# In[ ]:


# Aggregated Training Error
model = build_model()

checkpointer = ModelCheckpoint(filepath="/home/zg2309/model/{}.h5".format(model_name), verbose=1, save_best_only=True)
history = model.fit(train_X, train_y, validation_split=val_ratio, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpointer])


# In[ ]:


# Plot training & validation loss values
fig = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
fig.savefig('/home/zg2309/history/{}'.format(model_name))

score = model.evaluate(test_X, test_y, batch_size=batch_size)


# In[ ]:





# In[ ]:




