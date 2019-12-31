#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import random
import glob


# In[2]:


TRAIN_FOLDERS = '/home/dataset/augmented_training/*/'
TEST_FOLDERS = '/home/dataset/augmented_validation/*/'


train_folders = glob.glob(TRAIN_FOLDERS)
val_ratio = 0.1
train_folders_num = len(train_folders)

val_num = int(train_folders_num * val_ratio)
train_num = train_folders_num - val_num

random.shuffle(train_folders)

split_train_folders = train_folders[:train_num]
val_folders = train_folders[train_num:]
train_folders = split_train_folders

print('train folder num:', len(train_folders))
print('val folder num:', len(val_folders))


# In[3]:


test_folders = glob.glob(TEST_FOLDERS)
test_num = len(test_folders)
print('test folder num:', len(test_folders))


# In[4]:


# batch_size = 64
epochs = 10
n_hidden = 256
dim_input = 12

model_name = 'historical_lstm_epoch_{}_hidden_{}'.format(epochs, n_hidden)


# In[5]:


def train_generator():
    while True:
        idx = np.random.randint(0, train_num)
        folder = train_folders[idx]
        X = np.load(folder + 'X.npy')
        y = np.load(folder + 'y.npy')
        
        yield np.array([X]), np.array([y])
        
def val_generator():
    while True:
        idx = np.random.randint(0, val_num)
        folder = val_folders[idx]
        X = np.load(folder + 'X.npy')
        y = np.load(folder + 'y.npy')
        
        yield np.array([X]), np.array([y])
        


# In[6]:


def build_model():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(n_hidden,return_sequences=True, dropout=0.25,recurrent_dropout=0.1,input_shape=(None ,dim_input)))
    model.add(layers.LSTM(n_hidden, return_sequences=False, dropout=0.25,recurrent_dropout=0.1))
    model.add(layers.Dense(3))
#     adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    print(model.summary())
    return model


# In[7]:


# Aggregated Training Error
model = build_model()

history = model.fit_generator(train_generator(), steps_per_epoch=train_num, validation_steps=val_num, validation_data=val_generator(), epochs=epochs, verbose=1)


# In[ ]:


# Plot training & validation loss values
fig = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
fig.savefig('/home/zg2309/history/{}'.format(model_name))
plt.close(fig)

score = model.evaluate(test_X, test_y, batch_size=batch_size)


# In[ ]:


model.save("/home/zg2309/model/{}.h5".format(model_name))
print("Saved model to disk")


# In[ ]:




