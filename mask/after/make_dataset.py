from glob import glob

import cv2
import numpy as np
import random

class MakeDataset:
    def __init__(self, save=""):
        self.save = save

    # Brightness
    def chg_brightness(self, image, type):
        """
        :param image: original image
        :param type: "b" is brighter, "d" is darker
        :return: changed image
        """

        assert type in ["b", "d"], "please choose 'b' or 'd'"
        temp_array = np.full(image.shape, (128, 128, 128), dtype=np.uint8)

        if type == "b":
            chg_img = cv2.add(image, temp_array)
        elif type == "d":
            chg_img = cv2.subtract(image, temp_array)

        if self.save:
            cv2.imwrite(self.save, chg_img)

        return chg_img

    # Flip horizontal, vertical
    def flip_hroizontal(self, image, type):
        """
        :param image: original image
        :param type: "h" is horizontal, "v" is vertical
        :return: changed image
        """
        assert type in ["h", "v"], "please input 'h' or 'v'"

        if type == "h":
            flip_img = cv2.flip(image, 1)
        elif type == "v":
            flip_img = cv2.flip(image, 0)

        if self.save:
            cv2.imwrite(self.save, flip_img)

        return flip_img

    # move to horizontal
    def move_to_horizontal(self, image, px=0):
        pass

    # Rotation image
    def rotation_img(self, image, angle):
        pass


if __name__ == "__main__":
    img = cv2.imread('../face.jpg')

    md = MakeDataset("./test.jpg")
    # md.chg_brightness(img, "d")
    md.flip_hroizontal(img, "v")
    # md.move_to_horizontal(img, 0.2)