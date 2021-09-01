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

# In[3]:


train_dir = '/media/shreyashkharat/Storage Drive/Machine Learning, Deep Learning/Python/Projects/Image Classification Model/Color Data/data/training_set'
validation_dir = '/media/shreyashkharat/Storage Drive/Machine Learning, Deep Learning/Python/Projects/Image Classification Model/Color Data/data/validation_set'
test_dir = '/media/shreyashkharat/Storage Drive/Machine Learning, Deep Learning/Python/Projects/Image Classification Model/Color Data/data/test_set'


# Process for getting data from images:
# * Read the picture files.
# * Decode the JPEG into RGB grods of pixels.
# * Convert these into floating point tensors.
# * Rescale the pixel values into [0,1] intervals.

# In[4]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg


# In[5]:


train_data_idg = idg(rescale = 1./255)
test_data_idg = idg(rescale = 1./255)
train_generator = train_data_idg.flow_from_directory(train_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary')
validation_generator = test_data_idg.flow_from_directory(validation_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary')
# target_size is the resized size we want.


# In[6]:


from tensorflow.keras import layers
from tensorflow.keras import models


# In[7]:


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


# In[8]:


model_cnn.summary()


# In[9]:


from tensorflow.keras import optimizers
model_cnn.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(learning_rate = 0.00001), metrics = ['acc'])


# In[10]:


history = model_cnn.fit_generator(train_generator, steps_per_epoch = 100, epochs = 50, validation_data = validation_generator, validation_steps = 50)


# ## Performance Evaluation

# In[11]:


pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# * The accuracy is upto 0.7480.

# In[12]:


model_cnn.save('model_cnn.h5')
from tensorflow.keras import backend
backend.clear_session()


# ## Data Augmentation

# ### Preprocessing

# In[13]:


train_data_aug = idg(rescale = 1./255, rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data_aug = idg(rescale = 1./255, rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
train_gen_aug = train_data_aug.flow_from_directory(train_dir, target_size = (150, 150), batch_size = 32, class_mode = 'binary')
valid_gen_aug = test_data_aug.flow_from_directory(validation_dir, target_size = (150, 150), batch_size = 32, class_mode = 'binary')


# ### Model Architectue

# In[14]:


from tensorflow.keras import layers
from tensorflow.keras import models
model_cnn_aug = models.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
])
# The dropout layer drops given percentage of neurons to avoid over-fitting.


# In[15]:


model_cnn_aug.summary()


# In[16]:


from tensorflow.keras import optimizers
model_cnn_aug.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(learning_rate = 0.0001), metrics = ['acc'])


# In[17]:


history_aug = model_cnn_aug.fit_generator(train_gen_aug, steps_per_epoch = 100, epochs = 60, validation_data = valid_gen_aug, validation_steps = 50)


# In[18]:


pd.DataFrame(history_aug.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[19]:


model_cnn_aug.save('model_cnn_aug.h5')
from tensorflow.keras import backend
backend.clear_session()


# ## VGG16

# In[20]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg
train_data_vgg = idg(rescale = 1./255, rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data_vgg = idg(rescale = 1./255)
train_gen_vgg = train_data_vgg.flow_from_directory(train_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary')
validation_gen_vgg = test_data_vgg.flow_from_directory(validation_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary')


# In[21]:


from tensorflow.keras.applications import VGG16
vvg_convo = VGG16(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))


# In[22]:


from tensorflow.keras import models, layers
model_vgg = models.Sequential([
    vvg_convo,
    layers.Flatten(),
    layers.Dense(256, activation = 'relu'),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
])


# In[23]:


from tensorflow.keras import optimizers
model_vgg.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(learning_rate = 0.0002), metrics = ['acc'])


# In[24]:


checkpoints = keras.callbacks.ModelCheckpoint('model_vgg.h5')


# In[ ]:


history_vgg = model_vgg.fit_generator(train_gen_vgg, steps_per_epoch = 100, epochs = 40, validation_data = validation_gen_vgg, validation_steps = 50, callbacks = [checkpoints])


# In[ ]:


pd.DataFrame(history_vgg.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# * The above vgg model is not trianed due to high computational time (takes upto 8-10 mins per epoch).
# * So, the augmented CNN models gives best accuracy on validation set upto 0.7906.
