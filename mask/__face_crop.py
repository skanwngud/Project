import glob
import cv2
import numpy as np
import os

class FaceCrop:
    def __init__(self):
        self.__face_cascade_path = ""
        self.__eye_cascade_path = ""
        