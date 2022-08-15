from glob import glob
from PIL import Image, ImageEnhance, ImageChops

import cv2
import numpy as np
import random

class MakeDataset:
    def __init__(self, save=""):
        self.save = save

    # Brightness
    def chg_brightness(self, image, enhance_value):
        enhanceer = ImageEnhance.Brightness(image)
        birghtness_image = enhanceer.enhance(enhance_value)

        if self.save:
            birghtness_image.save(self.save)

        return birghtness_image

    # Flip horizaontal
    def flip_hroizontal(self, image):
        horizontal_flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.save:
            horizontal_flip_image.save(self.save)

        return horizontal_flip_image

    # FLip vertical
    def flip_vertical(self, image):
        vertical_flip_image = image.transpose(Image.FLIP_TOP_BOTTOM)

        if self.save:
            vertical_flip_image.save(self.save)

        return vertical_flip_image

    # move to horizontal
    def move_to_horizontal(self, image, px=0):
        width, height = image.size
        shift = random.randint(0, int(width * px))

        horizontal_shift_image = ImageChops.offset(image, shift, 0)
        horizontal_shift_image.paste((0), (0, 0, shift, height))

        if self.save:
            horizontal_shift_image.save(self.save)

        return horizontal_shift_image


if __name__ == "__main__":
    img = Image.open("../face.jpg")
    img = img.convert("RGB")

    md = MakeDataset("./test.jpg")
    # md.chg_brightness(img, 1.8, )
    # md.flip_hroizontal(img)
    md.move_to_horizontal(img, 0.2)