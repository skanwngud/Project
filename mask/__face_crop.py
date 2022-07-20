import glob
import cv2
import numpy as np
import os

from file_manager import FileManager

class FaceCrop:
    def __init__(self):
        super(FaceCrop).__init__()
        self.__face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.__eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

        self.face_cascade = cv2.CascadeClassifier(self.__face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(self.__eye_cascade_path)

        self.file_manager = FileManager()
        
    def find_face(self, img):
        face = self.face_cascade.detectMultiScale(img, 1.3, 5)
        return face
    
    def find_eye(self, img):
        eye = self.eye_cascade.detectMultiScale(img, 1.5, 5)
        return eye

    def crop_face(self, path):
        img_list = self.file_manager.get_images_list(path + "*.jpg")

        for img in img_list:
            image = self.file_manager.load_img(path + img)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.find_face(image_gray)

            if len(faces) != 0:
                for (x, y, w, h) in faces:
                    cropped = image[x:x+w, y:y+h]
                    cv2.imwrite(path + img.split('.')[0] + "_crop.jpg", cropped)

if __name__ == "__main__":
    face_crop = FaceCrop()
    face_crop.crop_face("./")

