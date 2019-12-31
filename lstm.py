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


# TRAIN_FILES = '/home/dataset/training*/*/data_smooth.csv'
# TEST_FILES = '/home/dataset/validation*/*/data_smooth.csv'
BOX_PTS = 9
TRAIN_FILES = '/home/dataset/team2/training/training*/*_smooth_{}.csv'.format(BOX_PTS)
TEST_FILES = '/home/dataset/team2/validation/validation*/*_smooth_{}.csv'.format(BOX_PTS)



train_csvs = glob.glob(TRAIN_FILES)

val_ratio = 0.1
train_csv_num = len(train_csvs)

val_num = int(train_csv_num * val_ratio)
train_num = train_csv_num - val_num

random.shuffle(train_csvs)

split_train_csvs = train_csvs[:train_num]
val_csvs = train_csvs[train_num:]
train_csvs = split_train_csvs

# print(train_csvs)
# print(val_csvs)
print('train csv num:', len(train_csvs))
print('val csv num:', len(val_csvs))


# In[3]:


test_csvs = glob.glob(TEST_FILES)
test_num = len(test_csvs)
# print(test_csvs)
print('test csv num:', len(test_csvs))


# In[4]:


max_frame = 200
dim_input = 12

output_dim = max_frame * 3

batch_size = 64
epochs = 300
n_hidden = 256

model_name = 'triple_lstm_epoch_{}_hidden_{}_team2'.format(epochs, n_hidden)

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(n_hidden,return_sequences=True, dropout=0.25,recurrent_dropout=0.1,input_shape=(max_frame ,dim_input)))
    model.add(layers.LSTM(n_hidden, return_sequences=True, dropout=0.25,recurrent_dropout=0.1))
    model.add(layers.LSTM(n_hidden, return_sequences=False, dropout=0.25,recurrent_dropout=0.1))
    model.add(layers.Dense(output_dim))
#     adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    print(model.summary())
    return model


# In[5]:


train_data = []
for file in train_csvs:
    raw = pd.read_csv(file).values
    length = raw.shape[0]
    padding = np.zeros((max_frame, dim_input + 3))
    padding[:length, :] = raw
    train_data.append(padding)
    
train_data = np.array(train_data)

train_X = train_data[:,:,:-3]
train_y = train_data[:,:,-3:]
train_y = train_y.reshape(-1, output_dim)

test_data = []
for file in test_csvs:
    raw = pd.read_csv(file).values
    length = raw.shape[0]
    padding = np.zeros((max_frame, dim_input + 3))
    padding[:length, :] = raw
    test_data.append(padding)

test_data = np.array(test_data)
test_X = test_data[:, :, :-3]
test_y = test_data[:, :, -3:]
test_y = test_y.reshape(-1, output_dim)

val_data = []
for file in val_csvs:
    raw = pd.read_csv(file).values
    length = raw.shape[0]
    padding = np.zeros((max_frame, dim_input + 3))
    padding[:length, :] = raw
    val_data.append(padding)

val_data = np.array(val_data)
val_X = val_data[:, :, :-3]
val_y = val_data[:, :, -3:]
val_y = val_y.reshape(-1, output_dim)

print("train X: ", train_X.shape)
print("train y: ", train_y.shape)

print("test X: ", test_X.shape)
print("test y: ", test_y.shape)

print("val X: ", val_X.shape)
print("val y: ", val_y.shape)


# In[6]:


# Aggregated Training Error
model = build_model()

history = model.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)


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




