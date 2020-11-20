def convolution(input, filter):
  init = 0
  for i in range(len(input)):
    for j in range(len(input[0])):
      init += input[len(input[0])-i-1][len(input)-j-1]*filter[i][j]
  return init


print(convolution([[5,10],[2,12]], [[1,-1],[2,-1]]))


from zipfile import ZipFile
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
import glob
import cv2
import numpy as np
from PIL import Image

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

with ZipFile('male-clothing.zip') as zipObj:
    zipObj.extractall()
with ZipFile('female-clothing.zip') as zipObj:
    zipObj.extractall()

size = (200,250)

mendata = glob.glob("Man's Clothing - n03746330/*")
womendata = glob.glob("Woman's Clothing - n04596852/*")

image_list = []
label_list = []

for filename in mendata:
    image = cv2.imread(filename)
    image = cv2.resize(image,(200,250))
    image_list.append(image)
    label_list.append(0)

for filename in womendata: #assuming gif
    image = cv2.imread(filename)
    image = cv2.resize(image,(200,250))
    image_list.append(image)
    label_list.append(1)

image_list = preprocess_input(np.array(image_list))
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size = 0.33)


model = VGG16(weights='imagenet', include_top=False)
features = model.predict(x_train)

VGG_model = Sequential()
VGG_model.add(Flatten(input_shape = features.shape[1:]))
VGG_model.add(Dense(256, activation='relu'))
VGG_model.add(Dropout(0.2))
VGG_model.add(Dense(1, activation='sigmoid'))

VGG_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

VGG_model.fit(features, y_train, epochs=20, batch_size=128)
