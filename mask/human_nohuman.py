# 필요 라이브러리 임포트
from glob import glob # 다량의 이미지를 불러오기 위한 라이브러리
from selenium import webdriver # chrome 사용하기 위한 webdriver
from selenium.webdriver.common.keys import Keys
import time # 페이지 로드를 위해 딜레이를 주기 위함
import urllib.request # url 이동

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # 이미지 로드를 위함

import cv2

import tensorflow
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, \
    BatchNormalization, Activation, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop

from sklearn.model_selection import train_test_split

# image_list=glob('c:/datasets/face_train/*.jpg')

data=list()
label=list()
img_list=glob('c:/datasets/face_train/human/*.jpg')
par_list=glob('c:/datasets/face_train/no_human/*.jpg') # 유사 사람 얼굴

for i in img_list:
    img=tf.keras.preprocessing.image.load_img(i,
    color_mode='rgb',
    target_size=(128, 128))
    img=np.array(img)/255.
    data.append(img)
    label.append(0)
    
for i in par_list:
    try:
        par=tf.keras.preprocessing.image.load_img(i,
        color_mode='rgb',
        target_size=(128, 128))
        par=np.array(par)/255.
        data.append(par)
        label.append(1)
    except:
        pass

data=np.array(data)
label=np.array(label)

# datagen=ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     vertical_flip=True,
#     horizontal_flip=True,
#     validation_split=0.2,
#     rescale=1./255
# )

# datagen2=ImageDataGenerator()

# print(type(data)) # 867
# print(type(label)) # 867
# print(len(data)) # 867
# print(len(label)) # 867
# print(len(img_list)) # 393
# print(len(par_list)) # 473

# print(data) 
# print(label)

# data = x
# label = y

x_train, x_test, y_train, y_test=train_test_split(
    data, label,
    train_size=0.95,
    random_state=23
)

x_train, x_val, y_train, y_val=train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=23
)

# trainset=datagen.flow(
#     x_train, y_train,
#     batch_size=32
# )

# valset=datagen2.flow(
#     x_val, y_val
# )

# testset=datagen2.flow(
#     x_test, y_test
# )

# print(x_train.shape) # (658, 128, 128, 3)

# x_train=x_train.reshape(-1, 128, 128, 1)
# x_test=x_test.reshape(-1, 128, 128, 1)
# x_val=x_val.reshape(-1, 128, 128, 1)

np.save('c:/datasets/face_train/x_train.npy', arr=x_train) # x
np.save('c:/datasets/face_train/y_train.npy', arr=y_train) # y
np.save('c:/datasets/face_train/x_val.npy', arr=x_val) # x
np.save('c:/datasets/face_train/y_val.npy', arr=y_val) # y
np.save('c:/datasets/face_train/x_test.npy', arr=x_test)
np.save('c:/datasets/face_train/y_test.npy', arr=y_test)


es=EarlyStopping(
    monitor='val_loss',
    patience=100,
    verbose=1
)

rl=ReduceLROnPlateau(
    monitor='val_loss',
    patience=30,
    verbose=1
)

mc=ModelCheckpoint(
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    filepath='c:/data/modelcheckpoint/project_{val_loss:.4f}_{val_acc:.4f}.hdf5'
)

model=Sequential()
model.add(Conv2D(64, 2, padding='same', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))

model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))

model.add(Conv2D(256, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Flatten())

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1, activation='sigmoid'))

# 컴파일, 훈련
model.compile(
    optimizer=RMSprop(
        learning_rate=0.1,
        epsilon=None),
    loss='binary_crossentropy',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=16,
    callbacks=[es, rl, mc]
)

# model.fit_generator(
#     trainset,
#     validation_data=valset,
#     steps_per_epoch=21,
#     epochs=1000,
#     callbacks=[es, rl, mc]
# )

loss=model.evaluate(
    x_test, y_test
)

# loss=model.evaluate_generator(
#     testset
# )

pred=model.predict(
    x_test
)

# pred=model.predict_generator(
#     x_test
# )

pred=np.where(pred>0.5, 1, 0)
pred1=pred[:5]

print(loss)
print(pred1)


# results
# [0.5822412967681885, 0.8181818127632141]
# [1]

# results
# [1.1617838144302368, 0.6818181872367859]

'''
for i in img_list:
    # img=cv2.imread('c:/datasets/face_train/human/face_train (%s).jpg'%i)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    img=np.array(img)/255.
    img_list.append(img)
    # label.append(0)


for i in par_list:
    par_img=cv2.imread('c:/datasets/face_train/no_human/pareidolia%s.jpg'%i)
    par_img=cv2.cvtColor(par_img, cv2.COLOR_BGR2RGB)
    par_img=cv2.resize(par_img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    par_img=np.array(par_img)/255.
    par_img.append(par_img)
    # label.append(1)

'''
'''

filepath='c:/datasets/face_train/'

import io

train=datagen.flow_from_directory(
    filepath,
    target_size=(256, 256),
    class_mode='sparse',
    subset='training',
    batch_size=696,
    shuffle=False
) # 696

# val=datagen.flow_from_directory(
#     filepath,
#     target_size=(256, 256),
#     class_mode='sparse',
#     subset='validation',
#     batch_size=173,
#     shuffle=False
# ) # 173


print(train[0])
print(val[0])
'''