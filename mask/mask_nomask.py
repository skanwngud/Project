import tensorflow
import numpy as np
import tensorflow as tf
import glob
import cv2
import datetime
import matplotlib.pyplot as plt
# import cvlib as cv

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dropout, Dense, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop, Adamax, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from PIL import Image

from sklearn.model_selection import train_test_split

# img_list=glob('c:/datasets/face_train/human/mask/*.jpg')
# img_list_2=glob('c:/datasets/face_train/human/nomask/*.jpg')

# 이미지 로드
img_list=glob.glob('c:/datasets/train_face/train_mask/*.jpg')
img_list_2=glob.glob('c:/datasets/train_face/train_nomask/*.jpg')
par_img=glob.glob('c:/datasets/train_face/pareidolia/*.jpg')


data=list()
label=list()

count=1
print('preprocessing')
str_time=datetime.datetime.now()
for i in img_list: # mask
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img=cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7, searchWindowSize=21)
        img=np.array(img)/255.
        cv2.imwrite('c:/datasets/face_train/train_mask/edit_' + str(count) + '.jpg', img)
        data.append(img)
        label.append(0)
    except:
        pass
print('mask preprocessing : ', datetime.datetime.now()-str_time) # 2 min

for i in img_list_2: # nomask
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img=cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7, searchWindowSize=21)
        img=np.array(img)/255.
        cv2.imwrite('c:/datasets/face_train/train_mask/edit_' + str(count) + '.jpg', img)

        data.append(img)
        label.append(1)
    except:
        pass
print('nomask preprocessing : ',datetime.datetime.now()-str_time) # 2 min

for i in par_img:
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img=cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7, searchWindowSize=21)
        img=np.array(img)/255.
        cv2.imwrite('c:/datasets/face_train/train_mask/edit_' + str(count) + '.jpg', img)
        data.append(img)
        label.append(2)
    except:
        pass
print('test preprocessing : ',datetime.datetime.now()-str_time) # 5 min

data=np.array(data)
label=np.array(label)
# test=np.array(test)

# data=data.reshape(-1, 256, 256, 1)

print(data.shape)
print(label.shape)

x_train, x_test, y_train, y_test=train_test_split(data, label, train_size=0.9, random_state=23)

es=EarlyStopping(
    monitor='val_loss',
    patience=150,
    verbose=1
)

rl=ReduceLROnPlateau(
    monitor='val_loss',
    patience=20,
    verbose=1,
    factor=0.1
)

mc=ModelCheckpoint(
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    filepath='c:/data/modelcheckpoint/project_{val_loss:.4f}_{val_acc:.4f}.hdf5'
)

model=Sequential()
model.add(Conv2D(64, 2, padding='same', input_shape=(256, 256, 3)))
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
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1, activation='softmax'))

# 컴파일, 훈련
model.compile(
    optimizer=RMSprop(
        learning_rate=0.01
        ),
    loss='sparse_categorical_crossentropy',
    metrics='acc'
)

model.fit(
x_train, y_train,
validation_split=0.2,
epochs=1000,
batch_size=8,
callbacks=[es, rl, mc]
)

loss=model.evaluate(
    x_test, y_test
)

pred=model.predict(
   x_test
)

print(loss)
print(pred[0])

# plt.imshow(test[0])
# plt.show()
