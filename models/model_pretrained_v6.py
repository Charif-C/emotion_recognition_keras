
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...


# In[2]:


import keras


# In[3]:


from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam, SGD
import numpy as np
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.applications import VGG16

# In[4]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import confusion_matrix


# In[5]:


def plot_confusion_matrix(y_true, y_pred, cmap=plt.cm.Blues, normalise=False, title='Confusion matrix'):
    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
    fig = plt.figure(figsize=(7,7))
    matplotlib.rcParams.update({'font.size': 16})
    ax  = fig.add_subplot(111)
    matrix = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(matrix)
    for i in range(0,7):
        for j in range(0,7):
            ax.text(j,i,cm[i,j],va='center', ha='center')
    # ax.set_title('Confusion Matrix')
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()


# In[6]:


# load data
X_train=np.load('images_train.npy')
X_train=np.stack((X_train,)*3, -1)
y_train=np.load('emotions_train.npy')
y_train=to_categorical(y_train)

X_val=np.load('images_val.npy')
X_val=np.stack((X_val,)*3, -1)
y_val=np.load('emotions_val.npy')
y_val=to_categorical(y_val)

X_test=np.load('images_test.npy')
X_test=np.stack((X_test,)*3, -1)
y_test=np.load('emotions_test.npy')
y_test=to_categorical(y_test)


# In[7]:


class_names= ['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']


# In[8]:


# parameters:
batch_size = 128
epochs = 100


# In[9]:


# setup info:
print ('X_train shape: ', X_train.shape) # (n_sample, 1, 48, 48)
print ('y_train shape: ', y_train.shape) # (n_sample, n_categories)
print ('X_val shape: ', X_val.shape) # (n_sample, 1, 48, 48)
print ('y_val shape: ', y_val.shape) # (n_sample, n_categories)
print ('X_test shape: ', X_test.shape) # (n_sample, 1, 48, 48)
print ('y_test shape: ', y_test.shape) # (n_sample, n_categories)
print ('  img size: ', X_train.shape[1], X_train.shape[2])
print ('batch size: ', batch_size)
print ('  nb_epoch: ', epochs)
print ('classes: ', class_names)

# In[11]:


# create the base pre-trained model
base_model = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(48,48,3))


# In[12]:


block4=base_model.get_layer('block4_pool')


# In[13]:


x = block4.output
x=Flatten(name='flatten')(x)
x=Dense(2048, activation='relu', kernel_regularizer=l2(0.000001),kernel_constraint=max_norm(4), name='fc6')(x)
x=BatchNormalization(axis=1)(x)
x=Dropout(0.8)(x)
x=Dense(2048, activation='relu', kernel_regularizer=l2(0.000001),kernel_constraint=max_norm(4), name='fc7')(x)
x=BatchNormalization(axis=1)(x)
pred=Dense(7, activation='softmax', kernel_regularizer=l2(0.000001),name='predictions')(x)


# In[15]:


# this is the model we will train
model = Model(inputs=base_model.input, outputs=pred)


# In[17]:


#base_model.layers[:-8]


# In[18]:


# first: train only the added classifier layers (which were randomly initialized)
#train the last block
for layer in base_model.layers[:-8]:
    layer.trainable = False


# In[19]:


model.summary()


# In[17]:


checkpoint = ModelCheckpoint('pretrainedv6.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')


# In[23]:


# compile/fit/train
# optimizer:
lr=0.00001
decay = lr/epochs

adam=Adam(lr=lr, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print ('Training....')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_val, y_val), shuffle=True, verbose=1, callbacks=[checkpoint])


# In[22]:


# model result:
loss_and_metrics = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=1)
print ('Done!')
print ('Loss : ', loss_and_metrics[0])
print ('Accuracy : ', loss_and_metrics[1])


# In[21]:


# save model
#model.save('model_pretrained.h5')


# In[ ]:


y_pred=model.predict(X_test, batch_size=batch_size, verbose=1, steps=None)


# In[38]:


y_pred_prob=[np.argmax(prob) for prob in y_pred]
y_true = [np.argmax(true) for true in y_test]


# In[ ]:


cf_mx=plot_confusion_matrix(y_true, y_pred_prob,cmap=plt.cm.YlGnBu, title='Confusion matrix, without normalization')
cf_mx_norm=plot_confusion_matrix(y_true, y_pred_prob,cmap=plt.cm.YlGnBu, normalise=True,title='Normalized confusion matrix')
