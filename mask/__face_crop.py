import glob
import cv2
import numpy as np
import os

from file_manager import FileManager

class FaceCrop:
    def __init__(self):
        super(FaceCrop).__init__()
        self.__face_cascade_path = "find_face.xml"
        self.__eye_cascade_path = "find_eye.xml"