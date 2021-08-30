#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


# In[2]:


import tensorflow as tf
from tensorflow import keras


# ## Data Preprocessing

# * The dataset we are going use contains 4000 images of cats and dogs, we will use 2000 images for training the model, 1000 images for validating the model, 1000 images for testing the model.

# In[4]:


train_dir = '/media/shreyashkharat/Storage Drive/Machine Learning, Deep Learning/Python/Projects/Image Classification Model/Color Data/data/train'
validation_dir = '/media/shreyashkharat/Storage Drive/Machine Learning, Deep Learning/Python/Projects/Image Classification Model/Color Data/data/validation'
test_dir = '/media/shreyashkharat/Storage Drive/Machine Learning, Deep Learning/Python/Projects/Image Classification Model/Color Data/data/test'


# Process for getting data from images:
# * Read the picture files.
# * Decode the JPEG into RGB grods of pixels.
# * Convert these into floating point tensors.
# * Rescale the pixel values into [0,1] intervals.

# In[6]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg


# In[8]:


train_data_idg = idg(rescale = 1./255)
test_data_idg = idg(rescale = 1./255)
train_generator = train_data_idg.flow_from_directory(train_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary')
validation_generator = test_data_idg.flow_from_directory(validation_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary')
# target_size is the resized size we want.


# In[9]:


from tensorflow.keras import layers
from tensorflow.keras import models


# In[11]:


model_cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
])


# In[13]:


model_cnn.summary()


# In[14]:


from tensorflow.keras import optimizers
model_cnn.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(learning_rate = 0.00001), metrics = ['acc'])


# In[17]:


history = model_cnn.fit_generator(train_generator, steps_per_epoch = 100, epochs = 30, validation_data = validation_generator, validation_steps = 50)


# In[18]:


pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# As the computation time for this model was pretty high(25-26 mins), we will save this model.

# In[20]:


model_cnn.save('image_classifier_cnn.h5')


# In[ ]:




