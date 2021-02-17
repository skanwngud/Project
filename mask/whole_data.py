import numpy as np
import glob
import tensorflow

import cv2

from keras.models import load_model

imglist=glob.glob('e:/datasets/face_train/all/*.jpg')

img_list=list()
for i in imglist:
    img=cv2.imread(i)
    img=cv2.resize(img, (256, 256))
    img=np.array(img)
    img_list.append(img)

img_list=np.array(img_list)
# img_list=img_list.reshape(-1, 256, 256, 1)

# print(img_list[0].shape) # (512, 512, 3)

x_train=np.load('c:/data/npy/pro_x_train.npy')
y_train=np.load('c:/data/npy/pro_y_train.npy')
x_test=np.load('c:/data/npy/pro_x_test.npy')
y_test=np.load('c:/data/npy/pro_y_test.npy')
x_val=np.load('c:/data/npy/pro_x_val.npy')
y_val=np.load('c:/data/npy/pro_y_val.npy')

model=load_model(
    'c:/data/modelcheckpoint/project.hdf5'
)

# loss=model.evaluate(x_test, y_test)
pred=model.predict(img_list)
pred=np.where(pred>0.5, 1, 0) # 1 은 사람, 0 은 파레이돌리아


print('전체 사진 : ', len(img_list))
print('사람 : ', np.count_nonzero(pred))
print('비율 : ', np.count_nonzero(pred)/len(img_list))

# print('loss : ', loss)


# results

# ideal
# about 0.7

# 전체 사진 :  2534
# 사람 :  1331
# 비율 :  0.5252565114443567
# loss :  [0.6427163481712341, 0.7045454382896423]

# project_0.6214_0.7452
# 