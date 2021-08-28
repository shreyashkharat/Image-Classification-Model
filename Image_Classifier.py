#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf
from tensorflow import keras


# The database used here is already in keras.

# In[3]:


fashion_data = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_data.load_data()


# In[4]:


plt.imshow(x_train[0])


# In[5]:


y_train[0]


# In[6]:


class_names = ['Tshirt', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'shirt', 'sneaker', 'bag', 'ankle boots']


# In[7]:


class_names[y_train[0]]


# ## Data Normalization

# In[8]:


x_train_normal = x_train/255
x_test_normal = x_test/255


# Here we need to divide data in 3 sets:
# * Training Set
# * Validation Set
# * Test set

# In[9]:


x_valid, x_train_f = x_train_normal[:5000], x_train_normal[5000:]
y_valid, y_train_f = y_train[:5000], y_train[5000:]


# In[10]:


x_test_f = x_test_normal


# ## Create the model architechture

# There are two APIs for defining model in Keras:
# * Sequential Model API
# * Functional API

# In[11]:


np.random.seed(42)
tf.random.set_seed(42)


# ![Image%20classifier%20model%20structure.png](attachment:Image%20classifier%20model%20structure.png)

# In[12]:


model_1 = keras.models.Sequential()
model_1.add(keras.layers.Flatten(input_shape = [28,28])) # input layer
# The number in argument of keras.....Dense() is the no. of neuron we want in that layer.
model_1.add(keras.layers.Dense(300, activation = 'relu')) # hidden layer1
model_1.add(keras.layers.Dense(100, activation = 'relu')) # hidden layer2
model_1.add(keras.layers.Dense(10, activation = 'softmax')) # Output Softmax layer


# In[13]:


model_1.summary()


# In[14]:


import pydot
keras.utils.plot_model(model_1)


# In[15]:


weights, biases = model_1.layers[1].get_weights()


# In[16]:


weights


# In[17]:


biases


# In[18]:


biases.shape


# ## Compling the model

# In[19]:


model_1.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
# loss is sparse cate....tropy as our y data is available in distinct labels.
# if we had probabilities we had to use categorical_crossentropy.
# if we had binary labels we have to use binary categorical crossentropy.
# for regression model use r2_score as accuracy. 


# In[20]:


model_1_history = model_1.fit(x_train_f, y_train_f, epochs = 50, validation_data = (x_valid, y_valid))


# In[21]:


model_1_history.params


# In[22]:


model_1_history.history


# In[23]:


# Plotting accuracy, loss, val_accuracy, val_loss
import pandas as pd
pd.DataFrame(model_1_history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# ### Performance Evaluation

# In[24]:


model_1.evaluate(x_test_f, y_test)


# In[ ]:




