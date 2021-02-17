import tensorflow
import numpy as np
import tensorflow as tf
import glob
import cv2
import datetime
import matplotlib.pyplot as plt
import cvlib as cv

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dropout, Dense, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop, Adamax, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

from sklearn.model_selection import train_test_split

# img_list=glob('c:/datasets/face_train/human/mask/*.jpg')
# img_list_2=glob('c:/datasets/face_train/human/nomask/*.jpg')

img_list=glob.glob('e:/datasets/train_face/train_mask/*.jpg')
img_list_2=glob.glob('e:/datasets/train_face/train_nomask/*.jpg')
test_img=glob.glob('e:/datasets/train_face/mask/*.jpg')


data=list()
label=list()
test=list()


# print('preprocessing')

for i in img_list: # mask
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img=cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7, searchWindowSize=21)
        cv2.imwrite('e:/datasets/train_face/train_mask/edit_mask'+str(i)+'.jpg', img)
        img=np.array(img)/255.
        data.append(img)
        label.append(0)
    except:
        pass

for i in img_list_2: # nomask
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img=cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7, searchWindowSize=21)
        cv2.imwrite('e:/datasets/train_face/train_mask/edit_nomask'+str(i)+'.jpg', img)
        img=np.array(img)/255.
        data.append(img)
        label.append(1)
    except:
        pass

for i in test_img:
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img=cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7, searchWindowSize=21)
        cv2.imwrite('e:/datasets/train_face/mask/edit_mask'+str(i)+'.jpg', img)
        img=np.array(img)/255.
        test.append(img)

x_train, x_val, y_train, y_val=train_test_split(
    data, label,
    train_size=0.8,
    random_state=23
)

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
model.add(Conv2D(64, 2, padding='same', input_shape=(256, 256, 1)))
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
        learning_rate=0.01,
        epsilon=None),
    loss='categorical_crossentropy',
    metrics='acc'
)

history=model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=12,
    callbacks=[es, rl, mc]
)

pred=model.predict(
    test
)

print(history[0])
print(history[1])

print(pred[0])

plt.imshow(test[0])
plt.show()