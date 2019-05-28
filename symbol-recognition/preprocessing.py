"""Data preprocessing."""
import skimage
import os


INPUT_PATH = "./data-augmented"
OUTPUT_PATH = "./data-augmented-preprocessed"


def mean_threshold(img: np.ndarray):
    """Apply thresholding on image."""
    threshold = img.mean()
    return img <= threshold


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """Transform image to grayscale and apply mean threshold."""
    res_img = skimage.color.rgb2gray(img)
    res_img = mean_threshold(res_img)
    return res_img


def read_class_data(class_path: str, inst_names: t.Iterable[str],
                    random_seed: int) -> np.ndarray:
    """Get image dataset from given ``filepath``."""
    CLASS_NAME = RE_CLASS_NAME.search(class_path).group()
    START_VAR_IND = len(inst_names)
    CLASS_FILEPATH = os.path.joint(OUTPUT_PATH, "_".join(("class", class_name)))

    if not os.path.exists(CLASS_FILEPATH):
        os.makedirs(CLASS_FILEPATH)

    for inst_name in inst_names:
        img_filepath = os.path.join(INPUT_PATH, img_name)

        img = skimage.io.imread(img_filepath)
        res_img = preprocess_img(img)

        res_img_filepath = os.path.join(OUTPUT_PATH, img_name)
        skimage.io.imwrite(res_img_filepath, res_img)


def preprocess() -> None:
    """Preprocess all training images."""
    file_tree = os.walk(dataset_path)
    file_tree.__next__()

    for dirpath, _, filenames in file_tree:
        read_class_data(
            class_path=dirpath,
            inst_names=filenames,
            random_seed=random_seed)


if __name__ == "__main__":
    preprocess()
