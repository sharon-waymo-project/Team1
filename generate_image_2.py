#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import shutil
import numpy as np
import glob
import cv2
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from tqdm import tqdm

tf.enable_eager_execution()

IMG_SIZE = 299
# In[2]:


def build_model():
    resnet = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    image_model = tf.keras.Model(inputs=resnet.input, outputs=resnet.layers[-2].output)
    return image_model

model = build_model()

VIEW = 2


# In[ ]:


train_path = '/home/dataset/training/*/*.tfrecord'

train_prefix = '/home/dataset/images/{}/'.format(VIEW)

if os.path.exists(train_prefix):
    shutil.rmtree(train_prefix)
os.mkdir(train_prefix)

train_prefix += 'training/'
os.mkdir(train_prefix)

train_tfs = glob.glob(train_path)

for file in tqdm(train_tfs):
    dataset = tf.data.TFRecordDataset(file, compression_type='')
    
    tar_name = file.split('/')[4]
    tar_path = train_prefix + tar_name + '/'
    
    if not os.path.exists(tar_path):
        os.mkdir(tar_path)
        
    seg_name = file.split('/')[5][:-9]
    seg_path = tar_path + seg_name + '/'
    
    if os.path.exists(seg_path):
        shutil.rmtree(seg_path)
    os.mkdir(seg_path)
    
    
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        image = frame.images[VIEW]
        img = tf.image.decode_jpeg(image.image).numpy()
        resize_img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        img_input = np.array([resize_img])
        img_embed = model.predict(img_input)
        img_embed = img_embed.squeeze()

        np.save(seg_path + '{}.npy'.format(idx), img_embed)



# In[ ]:


test_path = '/home/dataset/validation/*/*.tfrecord'
test_prefix = '/home/dataset/images/{}/'.format(VIEW)
test_prefix += 'validation/'
os.mkdir(test_prefix)


test_tfs = glob.glob(test_path)

for file in tqdm(test_tfs):
    dataset = tf.data.TFRecordDataset(file, compression_type='')
    
    tar_name = file.split('/')[4]
    tar_path = test_prefix + tar_name + '/'
    
    if not os.path.exists(tar_path):
        os.mkdir(tar_path)
        
    seg_name = file.split('/')[5][:-9]
    seg_path = tar_path + seg_name + '/'
    
    if os.path.exists(seg_path):
        shutil.rmtree(seg_path)
    os.mkdir(seg_path)
    
    
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        image = frame.images[VIEW]
        img = tf.image.decode_jpeg(image.image).numpy()
        resize_img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        img_input = np.array([resize_img])
        img_embed = model.predict(img_input)
        img_embed = img_embed.squeeze()
        np.save(seg_path + '{}.npy'.format(idx), img_embed)
        


# In[ ]:




