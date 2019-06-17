"""Data preprocessing."""
import typing as t
import os
import re

import numpy as np
import skimage
import imageio
import scipy.ndimage


OUTPUT_PATH = "./data-augmented-preprocessed"
RE_CLASS_NAME = re.compile(r"(?<=class_)[^_]+")
OUTPUT_FILE_TYPE = "png"


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """Transform image to grayscale and various transformation."""
    img = skimage.color.rgb2gray(img)

    # Mean thresholding
    img = img < img.mean()

    # Dilation
    img = scipy.ndimage.morphology.binary_dilation(img, iterations=2)

    # Empty border crop
    mask = np.argwhere(img)
    x_min, y_min = mask.min(axis=0)
    x_max, y_max = mask.max(axis=0) + 1
    img = img[x_min:x_max, y_min:y_max]

    # Resizing to fit CNN input shape
    img = skimage.transform.resize(
        image=img,
        output_shape=(45, 45),
        anti_aliasing=False,
        order=3)

    # Mean thresholding
    img = img >= img.mean()

    return img


def read_class_data(
        class_path: str,
        inst_names: t.Iterable[str]) -> np.ndarray:
    """Get image dataset from given ``filepath``."""
    CLASS_NAME = RE_CLASS_NAME.search(class_path).group()

    CLASS_FILEPATH_OUTPUT = os.path.join(
        OUTPUT_PATH, "_".join(("class", CLASS_NAME)))

    if not os.path.exists(CLASS_FILEPATH_OUTPUT):
        os.makedirs(CLASS_FILEPATH_OUTPUT)

    for img_name in inst_names:
        img_filepath = os.path.join(class_path, img_name)

        img = skimage.io.imread(img_filepath)
        res_img = preprocess_img(img)

        res_img_filepath = os.path.join(CLASS_FILEPATH_OUTPUT, img_name)

        imageio.imwrite(
            uri=res_img_filepath,
            im=res_img.astype(np.uint8),
            format=OUTPUT_FILE_TYPE)


def preprocess(dataset_path: str) -> None:
    """Preprocess all training images."""
    file_tree = os.walk(dataset_path)
    file_tree.__next__()

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for dirpath, _, filenames in file_tree:
        read_class_data(
            class_path=dirpath,
            inst_names=filenames)


if __name__ == "__main__":
    preprocess("./data-augmented")
