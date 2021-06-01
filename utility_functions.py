import pickle
from helper_functions import *
from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np
import os
from skimage import color


def resize_image(image, h=256, w=256, resample=3):
    return np.asarray(Image.fromarray(image).resize((h, w), resample=resample))


def preprocess_image(image):
    """ Resize and split image into L and ab channels. """
    image = resize_image(image)
    lab_image = color.rgb2lab(image)
    l_image = lab_image[:, :, 0]
    ab_image = lab_image[:, :, 1:]

    return np.round(l_image).astype(np.int8), np.round(ab_image).astype(np.int8)


def get_images(num_persons):
    all_people = os.listdir("./all_images")
    l_images = []
    ab_images = []
    all_people = all_people[:num_persons]

    print()
    print("--- Get images ---")
    for idx, image_on_person in enumerate(all_people):
        print(f"get image {idx}/{len(all_people)}")
        img_as_array = np.array(Image.open(f"./all_images/{image_on_person}"))
        l_img, ab_img = preprocess_image(img_as_array)
        l_images.append(l_img)
        ab_images.append(ab_img)
    l_images = np.array(l_images, dtype=object)
    ab_images = np.array(ab_images, dtype=object)
    return l_images, ab_images


def discretize_image(ab_image):
    return np.floor_divide(ab_image, 10) * 10


def get_color_key(a_color, b_color):
    return f"({a_color}, {b_color})"


class ColorizerTool:
    def __init__(self, images_limit=13233, height=256, width=256):
        self.ab_color_to_possible_color_idx = {}  # {"(3, 2)": 1, "(1, 1)": 0, "(2, 0)": 2}
        self.possible_color_idx_to_ab_color = {}  # {1: "(3, 2)", 0: "(1, 1)", 2: "(2, 0)"}
        self.height = height
        self.width = width
        self.images_limit = images_limit

    def set_possible_colors(self, save_files=False):
        all_people = os.listdir("./data/imageNet/")[:]
        color_index = 0

        for idx, image_on_person in enumerate(all_people):
            if idx % 50 == 49:
                print(f"set image {idx+1}/{self.images_limit}")

            rgb_image = np.array(Image.open(f"./data/imageNet/{image_on_person}"))

            ab_image = preprocess_image(rgb_image)[1]
            discretized_ab_image = discretize_image(ab_image)

            for y in range(self.height):
                for x in range(self.width):
                    a_color = discretized_ab_image[y][x][0]
                    b_color = discretized_ab_image[y][x][1]

                    color_key = get_color_key(a_color, b_color)

                    if self.ab_color_to_possible_color_idx.setdefault(color_key, color_index) is color_index:
                        color_index += 1

        self.possible_color_idx_to_ab_color = {v: k for k, v in self.ab_color_to_possible_color_idx.items()}

        if save_files:
            pickle.dump(self.ab_color_to_possible_color_idx, open("pickles/imageNet/ab_to_q_index_dict.p", "wb"))
            pickle.dump(self.possible_color_idx_to_ab_color, open("pickles/imageNet/q_index_to_ab_dict.p", "wb"))


def main():
    colorizer_tool = ColorizerTool()
    colorizer_tool.set_possible_colors(save_files=True)


if __name__ == "__main__":
    main()