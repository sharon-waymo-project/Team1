#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import random
import glob


# In[2]:


EPOCHS = 300
N_HIDDEN = 128
OBSERVE_LENGTH = 10
PREDICT_LENGTH = 5
FEAT_DIM = 12
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
IMG_DIM = 1536
VAL_RATIO = 0.1
TRAIN_FOLDERS = '/home/dataset/training_observe_{}_predict_{}/*/'.format(OBSERVE_LENGTH, PREDICT_LENGTH)
# TEST_FOLDERS = '/home/dataset/validation_observe_{}_predict_{}/*/'.format(OBSERVE_LENGTH, PREDICT_LENGTH)

MODEL_NAME = 'sslstm-image_epoch_{}_hidden_{}_observe_{}_predict_{}'.format(EPOCHS, N_HIDDEN, OBSERVE_LENGTH, PREDICT_LENGTH)


# In[3]:


train_folders = glob.glob(TRAIN_FOLDERS)
random.shuffle(train_folders)
val_num = int(VAL_RATIO * len(train_folders))
val_folders = train_folders[-val_num:]
train_folders = train_folders[:-val_num]

print('train folder num:', len(train_folders))
print('val folder num:', len(val_folders))


# In[4]:


import tensorflow.keras as keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, shuffle=True):
        'Initialization'
        self.batch_size = BATCH_SIZE
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_image = np.empty((self.batch_size, OBSERVE_LENGTH, IMG_DIM))
        X_feat = np.empty((self.batch_size, OBSERVE_LENGTH, FEAT_DIM))
        y = np.empty((self.batch_size, PREDICT_LENGTH, 3), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image_seq = []
            with open(ID + 'img_path_0.txt', 'r') as f:
                img_path = f.read().split('\n')

            for p in img_path:
                image_seq.append(np.load(p))
            
            X_image[i,] = np.array(image_seq)
            X_feat[i,] = np.load(ID + 'X.npy')[:, :-3]
            y[i] = np.load(ID + 'y.npy')

        return [X_image, X_feat], y


# In[5]:


def build_model():
    opt = optimizers.RMSprop(lr=LEARNING_RATE)
    
    
    feat_input = Input(shape=(OBSERVE_LENGTH, FEAT_DIM))
    img_input = Input(shape=(OBSERVE_LENGTH, IMG_DIM))
    
    #encoder_feat
    encoder_feat = layers.GRU(N_HIDDEN,
                  input_shape=(OBSERVE_LENGTH, FEAT_DIM),
                  return_sequences=False,
                  stateful=False,
                  dropout=0.2)(feat_input)

    encoder_img = layers.GRU(N_HIDDEN,
                  input_shape=(OBSERVE_LENGTH, IMG_DIM),
                  return_sequences=False,
                  stateful=False,
                  dropout=0.2)(img_input)
        
    concated = layers.concatenate([encoder_img, encoder_feat])
    
    rv = layers.RepeatVector(PREDICT_LENGTH)(concated)
    
    #lstm decoder
    decoder = layers.GRU(N_HIDDEN,
                  return_sequences=True,
                  stateful=False,
                  dropout=0.2)(rv)
    dense = layers.TimeDistributed(layers.Dense(3), input_shape=(PREDICT_LENGTH, None))(decoder)
    out = layers.Activation('linear')(dense)
    
    model = Model(inputs=[img_input, feat_input], outputs=[out])
    model.compile(loss='mse', optimizer=opt)
    
    print(model.summary())
    return model


# In[ ]:


# Aggregated Training Error
model = build_model()

# Generators
training_generator = DataGenerator(train_folders)
val_generator = DataGenerator(val_folders)

checkpointer = ModelCheckpoint(filepath="/home/zg2309/model/{}.h5".format(MODEL_NAME), verbose=1, save_best_only=True)
# early_stopping_callback = EarlyStopping(monitor='val_loss', mode='auto')
history = model.fit_generator(generator=training_generator,
                    validation_data=val_generator,
                    use_multiprocessing=True,
                    epochs=EPOCHS,
                    verbose=1,
                    workers=6,
                    callbacks=[checkpointer])
# history = model.fit(train_X, train_y, validation_split=val_ratio, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpointer])


# In[ ]:


# Plot training & validation loss values
fig = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
fig.savefig('/home/zg2309/history/{}'.format(MODEL_NAME))
plt.close(fig)


# In[ ]:





# In[ ]:




