"""Module dedicated to generate a balanced the data.

Both undersampling and oversampling are used: first, it is
calculated the trimmed mean (10% of cut) T of the size of
the classes. Then, every class greater than that value is
undersampled (randomly and uniformly), and every class
under that value is oversampled, both until every class
size is exactly T.
"""
import typing as t
import re
import os
import shutil

import scipy.stats
import numpy as np

OUTPUT_PATH = "./data-balanced"
RE_CLASS_NAME = re.compile(r"(?<=class_)[^_]+")
RE_OUTPUT_FORMAT = re.compile(r"(?<=\.).+$")


def undersampling(class_path: str, inst_names: t.Iterable[str],
                  class_name: str, random_seed: int, trimmed_class_size: int,
                  outpath: str) -> None:
    """Choose random instances of given class and copy to output path."""
    np.random.seed(random_seed)
    chosen_ind = np.random.choice(
        len(inst_names), size=trimmed_class_size, replace=False)

    for new_inst_ind, ind in enumerate(chosen_ind):
        chosen_inst = os.path.join(class_path, inst_names[ind])
        output_format = RE_OUTPUT_FORMAT.search(inst_names[ind]).group()

        inst_out_name = ".".join(
            ("_".join((class_name, str(ind))), output_format))

        shutil.copy(chosen_inst, os.path.join(outpath, inst_out_name))


def oversampling(class_path: str, inst_names: t.Iterable[str], class_name: str,
                 random_seed: int, trimmed_class_size: int,
                 outpath: str) -> None:
    """Choose random instances of given class and copy to output path."""
    # First, copy all original images to the output path
    for idx, inst in enumerate(inst_names):
        chosen_inst = os.path.join(class_path, inst)
        output_format = RE_OUTPUT_FORMAT.search(inst).group()

        shutil.copy(
            chosen_inst, os.path.join(
                outpath, "{}_{}.{}".format(class_name, idx, output_format)))

    # Then, oversampling randomly and uniformly
    class_size = len(inst_names)

    np.random.seed(random_seed)
    chosen_ind = np.random.choice(
        class_size, size=trimmed_class_size - class_size, replace=True)

    for new_inst_ind, ind in enumerate(chosen_ind):
        chosen_inst = os.path.join(class_path, inst_names[ind])
        output_format = RE_OUTPUT_FORMAT.search(inst_names[ind]).group()

        inst_out_name = ".".join(
            ("_".join((class_name, str(ind), str(new_inst_ind))),
             output_format))

        shutil.copy(chosen_inst, os.path.join(outpath, inst_out_name))


def read_class_data(class_path: str, inst_names: t.Iterable[str],
                    random_seed: int, trimmed_class_size: int) -> None:
    """Balance class in given ``class_path``."""
    CLASS_NAME = RE_CLASS_NAME.search(class_path).group()
    CLASS_FILEPATH = os.path.join(OUTPUT_PATH, "_".join(("class", CLASS_NAME)))

    if not os.path.exists(CLASS_FILEPATH):
        os.makedirs(CLASS_FILEPATH)

    class_size = len(inst_names)

    print(" Balancing class {} (size of {})..."
          "".format(CLASS_NAME, class_size))

    if class_size > trimmed_class_size:
        chosen_method = undersampling

    elif class_size < trimmed_class_size:
        chosen_method = oversampling

    else:
        return

    chosen_method(
        class_path=class_path,
        class_name=CLASS_NAME,
        inst_names=inst_names,
        random_seed=random_seed,
        trimmed_class_size=trimmed_class_size,
        outpath=CLASS_FILEPATH)


def class_trimmed_mean_size(dataset_path: str, cutoff: float = 0.1) -> int:
    """Get the trimmed mean of the class sizes."""
    file_tree = os.walk(dataset_path)
    file_tree.__next__()

    sizes = []
    for _, _, filenames in file_tree:
        sizes.append(len(filenames))

    return scipy.stats.trim_mean(sizes, proportiontocut=cutoff).astype(int)


def balance_classes(dataset_path: str, random_seed: int,
                    cutoff: float = 0.0) -> None:
    """Create a new directory with all classes balanced.

    The strategy adopted is a mix of undersampling and
    oversampling.
    """
    file_tree = os.walk(dataset_path)
    file_tree.__next__()

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    trimmed_class_size = class_trimmed_mean_size(
        dataset_path=dataset_path, cutoff=cutoff)

    print(" Balancing - trimmed_class_size (cut-off: {0}): {1}"
          "".format(cutoff, trimmed_class_size))

    for dirpath, _, filenames in file_tree:
        read_class_data(
            class_path=dirpath,
            inst_names=filenames,
            trimmed_class_size=trimmed_class_size,
            random_seed=random_seed)


if __name__ == "__main__":
    balance_classes("./data-original", random_seed=1234)
