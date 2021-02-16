import numpy as np
import glob
import tensorflow

from keras.models import load_model

imglist=glob.glob('c:/datasets/face/*.jpg')

img_list=list()
for i in imglist:
    try:
        img=tensorflow.keras.preprocessing.image.load_img(
            i,
            target_size=(512, 512),
            color_mode='rgb'
        )
        img=np.array(img)
        img_list.append(img)
    except:
        pass

print(len(img_list)) # 1487
print(img_list[0].shape) # (512, 512, 3)

# x_train=np.load('c:/datasets/face_train/x_train.npy')
# y_train=np.load('c:/datasets/face_train/y_train.npy')
# x_test=np.load('c:/datasets/face_train/x_test.npy')
# y_test=np.load('c:/datasets/face_train/y_test.npy')
# x_val=np.load('c:/datasets/face_train/x_val.npt.npy')
# y_val=np.load('c:/datasets/face_train/y_val.npy')

model=load_model(
    'c:/data/modelcheckpoint/project_0.5528_0.7273.hdf5'
)

model.summary()

# loss=model.evaluate(x_test, y_test)
pred=model.predict(img_list)
pred=np.where(np>0.5, 1, 0) # 1 은 사람, 0 은 파레이돌리아

# print(loss)
print(pred)

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 128, 128, 64)      832
_________________________________________________________________
batch_normalization (BatchNo (None, 128, 128, 64)      256
_________________________________________________________________
activation (Activation)      (None, 128, 128, 64)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 128, 128, 64)      16448
_________________________________________________________________
batch_normalization_1 (Batch (None, 128, 128, 64)      256
_________________________________________________________________
activation_1 (Activation)    (None, 128, 128, 64)      0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 128, 128, 64)      16448
_________________________________________________________________
batch_normalization_2 (Batch (None, 128, 128, 64)      256
_________________________________________________________________
activation_2 (Activation)    (None, 128, 128, 64)      0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 64, 64)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 64, 64, 128)       32896
_________________________________________________________________
batch_normalization_3 (Batch (None, 64, 64, 128)       512
_________________________________________________________________
activation_3 (Activation)    (None, 64, 64, 128)       0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 64, 64, 128)       65664
_________________________________________________________________
batch_normalization_4 (Batch (None, 64, 64, 128)       512
_________________________________________________________________
activation_4 (Activation)    (None, 64, 64, 128)       0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 64, 64, 128)       65664
_________________________________________________________________
batch_normalization_5 (Batch (None, 64, 64, 128)       512
_________________________________________________________________
activation_5 (Activation)    (None, 64, 64, 128)       0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 128)       0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 32, 32, 256)       131328
_________________________________________________________________
batch_normalization_6 (Batch (None, 32, 32, 256)       1024
_________________________________________________________________
activation_6 (Activation)    (None, 32, 32, 256)       0
_________________________________________________________________
flatten (Flatten)            (None, 262144)            0
_________________________________________________________________
dense (Dense)                (None, 1024)              268436480
_________________________________________________________________
batch_normalization_7 (Batch (None, 1024)              4096
_________________________________________________________________
activation_7 (Activation)    (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 1025
=================================================================
Total params: 268,774,209
Trainable params: 268,770,497
Non-trainable params: 3,712
_________________________________________________________________
'''