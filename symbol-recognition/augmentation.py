"""Auxiliary module dedicated to perform data augmentation."""
import typing as t
import os
import re

import imageio
import keras.preprocessing.image
import numpy as np


OUTPUT_PATH = "./data-augmented"
RE_CLASS_NAME = re.compile(r"(?<=class_)[^_]+")
OUTPUT_FILE_TYPE = "png"


def gen_variants(
        image: np.ndarray,
        random_seed: int) -> t.Sequence[np.ndarray]:
    """Generate image variants using random data augmentation."""
    return [image.copy() for _ in range(3)]


def write_variants(
        variants: t.Sequence[np.ndarray],
        class_name: str,
        start_var_ind: int) -> None:
    """Write image variants into output files."""
    for var_ind, var_img in enumerate(variants, start_var_ind):
        cur_var_filename = "_".join((class_name, str(var_ind)))
        cur_var_filepath = os.path.join(OUTPUT_PATH, cur_var_filename)

        imageio.imwrite(
            uri=".".join((cur_var_filepath, OUTPUT_FILE_TYPE)),
            im=var_img,
            format=OUTPUT_FILE_TYPE)


def read_class_data(
        class_path: str,
        inst_names: t.Iterable[str],
        random_seed: int) -> np.ndarray:
    """Get image dataset from given ``filepath``."""
    CLASS_NAME = RE_CLASS_NAME.search(class_path).group()
    START_VAR_IND = len(inst_names)

    for inst_name in inst_names:
        cur_inst = imageio.imread(os.path.join(class_path, inst_name))
        cur_vars = gen_variants(image=cur_inst, random_seed=random_seed)
        write_variants(
            variants=cur_vars,
            class_name=CLASS_NAME,
            start_var_ind=START_VAR_IND)


def augment_data(dataset_path: str, random_seed: int) -> None:
    """Augment data class by class."""
    file_tree = os.walk(dataset_path)
    file_tree.__next__()

    for dirpath, _, filenames in file_tree:
        read_class_data(class_path=dirpath,
                        inst_names=filenames,
                        random_seed=random_seed)


if __name__ == "__main__":
    augment_data("./data-original", random_seed=2805)
