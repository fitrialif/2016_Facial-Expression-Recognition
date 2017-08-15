# -*- coding: utf-8 -*-
"""
CS50 final project
UseCNN.py
"""

from keras import backend as K
K.set_image_dim_ordering('th')
import os
from PIL import Image
import numpy as np
from keras.models import model_from_json

#%%
'''
set variables
'''

# image size and parameters
img_rows, img_cols = 48,48
max_image = 50
rgb_max = 255

# batch size
batch_size = 128

#%%
'''
load images and preprocess
'''

# prepare for loading
img = []
size = (img_rows, img_cols)
data = np.empty((max_image,1,img_rows, img_cols),dtype="float32")
num = 0

# load images, resize images and convert to grey-scale images
imgs = os.listdir("./image")
num = len(imgs)
for i in range(num):     
    img = Image.open("./image/"+imgs[i])
    img = img.convert("L")
    img = img.resize(size)
    arr = np.asarray(img,dtype="float32")
    data[i,:,:,:] = arr

# load image data into proper shape
norm_data = np.asarray(data, dtype='float64')/rgb_max
flatten_data=np.empty((num,img_rows*img_cols))
for i in range(num):
    flatten_data[i]=np.ndarray.flatten(norm_data[i,0:img_rows,0:img_cols])

faces = flatten_data.reshape(flatten_data.shape[0],1,img_rows,img_cols)

#%%
'''
load model and pridict
'''

# open loaded model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
 
# compile loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

# predict images
result = loaded_model.predict_classes(faces, batch_size=batch_size, verbose=1)

# print results
number=0
print (result)
for i in range(result.size):
    print("The expression of image " + str(number) + " is:")
    if(result[i] == [0]): print("ANGRY!(◣_◢)")
    elif(result[i] == [1]): print("DISGUST! ewww~~")
    elif(result[i] == [2]): print("FEAR! Σ( ° △ °|||)︴")
    elif(result[i] == [3]): print("HAPPY!~(￣▽￣)~* ")
    elif(result[i] == [4]): print("SAD! (╥_╥)")
    elif(result[i] == [5]): print("SURPRISE!（⊙o⊙）")
    else: print("NEUTRAL :|")
    number+=1

