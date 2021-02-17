import tensorflow
import numpy as np
import tensorflow as tf
import glob
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dropout, Dense, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop, Adamax, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

from sklearn.model_selection import train_test_split

img_list=glob('c:/datasets/face_train/human/mask/*.jpg')
img_list_2=glob('c:/datasets/face_train/human/nomask/*.jpg')

data=list()
label=list()
for i in img_list:
    img=cv2.imread(i)
    img=cv2.resize(img, (256, 256))
    img=np.array(img)/255.
    data.append(img)
    label.append(0)

for i in img_list_2:
    img=cv2.imread(i)
    img=cv2.resize(img, (256, 256))
    img=np.array(img)/255.
    data.append(img)
    label.append(1)

