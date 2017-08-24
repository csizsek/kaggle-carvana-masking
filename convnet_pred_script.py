
# coding: utf-8

# In[1]:


# inspired by this:
# https://www.kaggle.com/bguberfain/naive-keras-2


# In[2]:


import base64
import collections
import copy
import cProfile
import datetime
import itertools
import json
import math
import os
import operator
import pickle
import random
import re
import shutil
import sys
import time

import cv2
import Image
import keras
from keras import *
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL as pil
import pylab
import scipy
import seaborn as sns
import skimage
import sklearn
from sklearn import *
import statsmodels as sm
import tensorflow as tf
import tqdm

np.random.seed(1337)


# In[3]:


data_dir = '/media/ntfs/data/carvana_masking'

input_dir = data_dir + '/input'
tmp_dir = data_dir + '/tmp'
output_dir = data_dir + '/output'

train_dir = input_dir + '/train'
train_mask_dir = input_dir + '/train_masks'

val_dir = input_dir + '/validation'
val_mask_dir = input_dir + '/validation_masks'

test_dir = input_dir + '/test'


# In[4]:


angle = int(sys.argv[1])

resize_h = 160
resize_w = 240

n_train = 256  # max: 257
n_val = 16     # max: 61

batch_size = 16

epochs = 100

smooth = 1.0


# In[5]:


train_ids = list(set([name[:12] for name in os.listdir(train_dir)]))
random.shuffle(train_ids)
train_ids = train_ids[:n_train]

val_ids = list(set([name[:12] for name in os.listdir(val_dir)]))
random.shuffle(val_ids)
val_ids = val_ids[:n_val]


# In[6]:


def load_image(id, angle, val=False):
    
    dir = train_dir
    if val:
        dir = val_dir
        
    img = image.load_img(
            dir + '/{0}_{1:02d}.jpg'.format(id, angle),
            target_size=(resize_h, resize_w))
    img = image.img_to_array(img)
    return img

def load_mask(id, angle, val=False):
    dir = train_mask_dir
    if val:
        dir = val_mask_dir
        
    img = image.load_img(
            dir + '/{0}_{1:02d}_mask.gif'.format(id, angle),
            target_size=(resize_h, resize_w))
    img = image.img_to_array(img)
    img = img[:,:,0]
    return img


# In[7]:


def load_data(n, angle=angle, val=False):
    
    dir = train_dir
    mask_dir = train_mask_dir
    ids = train_ids
    if val:
        dir = val_dir
        mask_dir = val_mask_dir
        ids = val_ids
    
    X = np.empty((n, resize_h, resize_w, 12), dtype=np.float32)
    y = np.empty((n, resize_h, resize_w, 1), dtype=np.float32)
    
    with tqdm.tqdm_notebook(total=n) as bar:
        
        for i in range(n):
            
            imgs = [load_image(ids[i], a, val) for a in range(1, 17)]
            X[i, ..., :9] = np.concatenate([imgs[angle-1], np.mean(imgs, axis=0), np.std(imgs, axis=0)], axis=2)
            y[i] = np.expand_dims(load_mask(ids[i], angle, val), 2) / 255.
            
            del imgs
            
            bar.update()
            
    return X, y


# In[8]:


X_train, y_train = load_data(n_train)
X_val, y_val = load_data(n_val, val=True)


# In[9]:


y_train_mean = y_train.mean(axis=0)
y_train_std = y_train.std(axis=0)
y_train_min = y_train.min(axis=0)

y_features = np.concatenate([y_train_mean, y_train_std, y_train_min], axis=2)

X_train[:, ..., -3:] = y_features
X_val[:, ..., -3:] = y_features


# In[10]:


X_mean = X_train.mean(axis=(0,1,2), keepdims=True)
X_std = X_train.std(axis=(0,1,2), keepdims=True)

X_train -= X_mean
X_train /= X_std

X_val -= X_mean
X_val /= X_std


# In[12]:


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# # Adding More Layers to Bruno's Model

# In[13]:


inp = keras.layers.Input((resize_h, resize_w, 12))

conv1 = keras.layers.Conv2D(12, 3, activation='relu', padding='same')(inp)
conv2 = keras.layers.Conv2D(12, 3, activation='relu', padding='same')(inp)
max1 = keras.layers.MaxPooling2D(2)(conv2)

conv3 = keras.layers.Conv2D(12, 3, activation='relu', padding='same')(max1)
conv4 = keras.layers.Conv2D(12, 3, activation='relu', padding='same')(conv3)
max2 = keras.layers.MaxPooling2D(2)(conv4)

conv5 = keras.layers.Conv2D(12, 3, activation='relu', padding='same')(max2)

deconv4 = keras.layers.Conv2DTranspose(12, 3, strides=2, activation='relu', padding='same')(conv4)
deconv5 = keras.layers.Conv2DTranspose(12, 3, strides=4, activation='relu', padding='same')(conv5)

deconvs = keras.layers.concatenate([conv2, deconv4, deconv5])

out = keras.layers.Conv2D(1, 7, activation='sigmoid', padding='same')(deconvs)

model = keras.models.Model(inp, out)

model.compile(keras.optimizers.Adam(lr=1e-3), dice_loss, metrics=['accuracy', dice_coef])


# In[14]:


history = model.fit(
            x=X_train,
            y=y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            verbose=0)


# # Predict

# In[19]:


def load_test_image(id, angle):
    
    dir = test_dir
        
    img = image.load_img(
            dir + '/{0}_{1:02d}.jpg'.format(id, angle),
            target_size=(resize_h, resize_w))
    img = image.img_to_array(img)
    return img


# In[20]:


# from here:
# https://www.kaggle.com/stainsby/fast-tested-rle

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


# In[21]:


test_ids = list(set([name[:12] for name in os.listdir(test_dir)]))#[:10]
n_test = len(test_ids)

with tqdm.tqdm_notebook(total=n_test) as bar:

    with open('pred_rle_{0:02d}.csv'.format(angle), 'w') as f:
        for id in test_ids:

            x = np.empty((resize_h, resize_w, 12), dtype=np.float32)

            imgs = [load_test_image(id, a) for a in range(1, 17)]

            x[..., :9] = np.concatenate([imgs[angle-1], np.mean(imgs, axis=0), np.std(imgs, axis=0)], axis=2)
            del imgs

            x[..., -3:] = y_features

            x -= X_mean[0]
            x /= X_std[0]

            y_pred = model.predict(x[None]).squeeze()
            del x

            y_pred_rgb = np.zeros((resize_h, resize_w, 3))
            y_pred_rgb[..., 0] = y_pred
            y_pred_rgb[..., 1] = y_pred
            y_pred_rgb[..., 2] = y_pred

            y_pred_resized = (image.img_to_array(
                image.array_to_img(y_pred_rgb).resize((1918, 1280), Image.BILINEAR))[..., 0] > 0.5)
            del y_pred
            del y_pred_rgb

            f.write('{0}_{1:02d}.jpg,{2}\n'.format(id, angle, rle_to_string(rle_encode(y_pred_resized))))

            del y_pred_resized

            bar.update()

