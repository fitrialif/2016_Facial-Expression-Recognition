# -*- coding: utf-8 -*-
"""
CS50 final project
TrainCNN.py
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np

#%%
'''
set variables
'''
dataset_path = 'fer2013.csv'

# batch size
batch_size = 128

# number of expression types
nb_classes = 7

# training epoch
nb_epoch = 50

# image size and parameters
img_rows, img_cols = 48,48
rgb_max = 255

# CNN parameters
nb_filters = 32
nb_pool = 2
nb_conv = 3

val_size = 3500


#%%
'''
load data from dataset
'''

# get images and labels from csv file
lines = [line.split(',') for line in open(dataset_path)]
faces = [[float(x) for x in line[1 : img_rows*img_cols+1]] for line in lines[0 : val_size*10]]
              
label=np.empty(val_size * 10)
i = 0
for line in lines[0 : val_size * 10]:
    label[i] = float(line[0])
    i += 1
label=label.astype(np.int)

# allocate train data and test data to make sure of randomness
X_train=np.empty((val_size * 9, img_rows * img_cols))
y_train=np.empty(val_size * 9)
X_test=np.empty((val_size, img_rows * img_cols))
y_test=np.empty(val_size)

for i in range(val_size):
    X_train[i * 9 : i * 9+ 9]=faces[i * 10 : i * 10 + 9]
    y_train[i * 9 : i * 9+ 9]=label[i * 10 : i * 10 + 9]
    X_test[i]=faces[i * 10 + 9]
    y_test[i]=label[i * 10 + 9]

# reshape data
X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
X_test = X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= rgb_max
X_test /= rgb_max

print('X_train shape:', X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')

Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)


#%%
'''
build CNN
'''
model = Sequential()

# first convolution2D layer
model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='valid',input_shape=(1,img_rows,img_cols)))
model.add(Activation('relu'))

# second convolution2D layer and maxpooling layer
model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
model.add(Dropout(0.5))

# add fully connected layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# add classifier layer
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# compile model
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

#%%
'''
fit model
'''
model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,
          verbose=1,validation_data=(X_test,Y_test))

#%%
'''
save model
'''
# sace model to json file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")





