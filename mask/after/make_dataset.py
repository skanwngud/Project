from glob import glob

import cv2
import numpy as np
import random

class MakeDataset:
    def __init__(self, save=""):
        self.save = save

    # Brightness
    def chg_brightness(self, image, enhance_value):
        pass

    # Flip horizaontal
    def flip_hroizontal(self, image):
        pass

    # FLip vertical
    def flip_vertical(self, image):
        pass

    # move to horizontal
    def move_to_horizontal(self, image, px=0):
        pass

if __name__ == "__main__":
    img = Image.open("../face.jpg")
    img = img.convert("RGB")

    md = MakeDataset("./test.jpg")
    # md.chg_brightness(img, 1.8, )
    # md.flip_hroizontal(img)
    md.move_to_horizontal(img, 0.2)