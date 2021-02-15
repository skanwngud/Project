# 필요 라이브러리 임포트
from glob import glob # 다량의 이미지를 불러오기 위한 라이브러리
from selenium import webdriver # chrome 사용하기 위한 webdriver
from selenium.webdriver.common.keys import Keys
import time # 페이지 로드를 위해 딜레이를 주기 위함
import urllib.request # url 이동

import numpy as np
from PIL import Image as pil

import tensorflow

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, \
    BatchNormalization, Activation, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

img_list=list()
for i in range(1, 513):
    filepath='c:/datasets/face_train/no_mask (%s).jpg'%i
    img=pil.open(filepath)
    img=np.array(img)
    img_list.append(img)

img_list=np.array(img_list)/255.

# print(img_list.shape) # (512, )
# print(img_list[0])
# print(img_list[0].shape) # (1000, 1000, 3)

img_train, img_val=train_test_split(
    img_list,
    train_size=0.9,
    random_state=23
)

# print(img_train.shape) # (460, )
# print(img_val.shape) # (52, )

img_train=np.resize(
    img_train,
    (460, 256, 256, 3)
)

img_val=np.resize(
    img_val,
    (52, 256, 256, 3)
)

# print(img_train.shape) # (460, 256, 256, 3)
# print(img_val.shape) # (52, 256, 256, 3)

model=Sequential()
model.add(Conv2D(128, 3, padding='same', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2, 'softmax'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics='acc'
)

model.fit(
    img_train

)