"""Data preprocessing."""
import typing as t
import os
import re

import numpy as np
import skimage
import imageio


INPUT_PATH = "./data-augmented"
OUTPUT_PATH = "./data-augmented-preprocessed"
RE_CLASS_NAME = re.compile(r"(?<=class_)[^_]+")
OUTPUT_FILE_TYPE = "png"


def mean_threshold(img: np.ndarray):
    """Apply thresholding on image."""
    threshold = img.mean()
    return 255 * (img <= threshold)


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """Transform image to grayscale and apply mean threshold."""
    res_img = skimage.color.rgb2gray(img)
    res_img = skimage.transform.resize(res_img, (32, 32), preserve_range=True)
    res_img = mean_threshold(res_img)

    return res_img


def read_class_data(
        class_path: str,
        inst_names: t.Iterable[str]) -> np.ndarray:
    """Get image dataset from given ``filepath``."""
    CLASS_NAME = RE_CLASS_NAME.search(class_path).group()

    CLASS_FILEPATH_INPUT = os.path.join(
        INPUT_PATH, "_".join(("class", CLASS_NAME)))
    CLASS_FILEPATH_OUTPUT = os.path.join(
        OUTPUT_PATH, "_".join(("class", CLASS_NAME)))

    if not os.path.exists(CLASS_FILEPATH_OUTPUT):
        os.makedirs(CLASS_FILEPATH_OUTPUT)

    for img_name in inst_names:
        img_filepath = os.path.join(CLASS_FILEPATH_INPUT, img_name)

        img = skimage.io.imread(img_filepath)
        res_img = preprocess_img(img)

        res_img_filepath = os.path.join(CLASS_FILEPATH_OUTPUT, img_name)

        imageio.imwrite(
            uri=res_img_filepath,
            im=res_img.astype(np.uint8),
            format=OUTPUT_FILE_TYPE)


def preprocess() -> None:
    """Preprocess all training images."""
    file_tree = os.walk(INPUT_PATH)
    file_tree.__next__()

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for dirpath, _, filenames in file_tree:
        read_class_data(
            class_path=dirpath,
            inst_names=filenames)


if __name__ == "__main__":
    preprocess()
