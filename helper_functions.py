from PIL import Image
import numpy as np
import os
from skimage import color


def resize_image(image, h=256, w=256, resample=3):
    return np.asarray(Image.fromarray(image).resize((h, w), resample=resample))


def preprocess_image(image):
    """
    Resize and split image into L and ab channels.
    :param image: RGB-image
    :return:
    """
    image = resize_image(image)
    lab_image = color.rgb2lab(image)
    l_image = lab_image[:, :, 0]
    ab_image = lab_image[:, :, 1:]
    # l_image_tensor = torch.Tensor(l_image)[None, None, :, :]
    # ab_image_tensor = torch.Tensor(ab_image)[None, None, :, :]
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


def get_ab_colors_from_key(color_key):
    splitted_colors = color_key.split(",")
    a_color = int(splitted_colors[0][1:])
    b_color = int(splitted_colors[1][1:-1])

    return a_color, b_color


def get_color_key(a_color, b_color):
    return f"({a_color}, {b_color})"


def print_dict(dict):
    for key in dict:
        print(f"{key} = {dict[key]}")


