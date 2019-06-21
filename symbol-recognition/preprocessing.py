"""Data preprocessing."""
import typing as t
import sys
import os
import re

import numpy as np
import skimage
import imageio

sys.path.insert(0, "../antideriv")
import preprocess as antideriv_preproc # noqa: ignore


OUTPUT_PATH = "./data-augmented-preprocessed"
RE_CLASS_NAME = re.compile(r"(?<=class_)[^_]+")
OUTPUT_FILE_TYPE = "png"

PREPROCESSOR_MODEL = antideriv_preproc.Preprocessor()
"""Preprocess the training data the same way as a regular input."""


def resize(img: np.ndarray,
           output_shape: t.Tuple[int, int] = (45, 45)) -> np.ndarray:
    """Resize image to ``output_shape`` with interpolation of order 3."""
    img = skimage.transform.resize(
        image=img,
        output_shape=output_shape,
        anti_aliasing=False,
        order=3)

    return img


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """Transform image to grayscale and various transformation."""
    img = skimage.color.rgb2gray(img)

    img = img < img.mean()
    img = np.pad(img, 4, mode="constant", constant_values=0)
    img = skimage.filters.rank.mean(img, selem=np.ones((2, 2)))

    # Preprocess the image the same way as a regular Antideriv input
    # except for the border cropping
    img = PREPROCESSOR_MODEL.preprocess(img, crop_borders=False)

    # Resize the image to a 45x45 image
    img = resize(img)

    # Mean thresholding to binarize the resized image
    img = img >= img.mean()

    return img.astype(np.uint8)


def read_class_data(
        class_path: str,
        inst_names: t.Iterable[str]) -> np.ndarray:
    """Get image dataset from given ``filepath``."""
    CLASS_NAME = RE_CLASS_NAME.search(class_path).group()

    CLASS_FILEPATH_OUTPUT = os.path.join(
        OUTPUT_PATH, "_".join(("class", CLASS_NAME)))

    if not os.path.exists(CLASS_FILEPATH_OUTPUT):
        os.makedirs(CLASS_FILEPATH_OUTPUT)

    print(" Preprocessing class {}...".format(CLASS_NAME))

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
