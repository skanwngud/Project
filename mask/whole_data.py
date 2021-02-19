import tensorflow
import tensorflow as tf
import numpy as np
import cv2


from glob import glob

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model=load_model('c:/data/modelcheckpoint/best_project.hdf5')

img_list=glob('c:/dataset/mask/export/images/*.jpg')

test=list()

datagen=ImageDataGenerator()

for i in img_list:
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=np.array(img)/255.
        test.append(img)
    except:
        pass

test=np.array(test)
test=datagen.flow()

pred=model.predict_generator(test)
pred=np.argmax(test, axis=-1)

print(type(pred[0]))
print('전체 : ', len(img_list)) # 1632
print('마스크사람 : ', len(img_list)-np.count_nonzero(pred))
print('마스크비율 : ', np.count_nonzero(pred)/len(img_list))
print(pred[0])

# results
# 전체 :  1009
# 마스크사람 :  -2018
# 마스크비율 :  3.0
# [0.37070063 0.5508767  0.07842268]