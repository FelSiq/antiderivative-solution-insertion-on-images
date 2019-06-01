"""Module dedicated to train the CNN model to object recognition."""
import typing as t
import inspect
import os
import re

import numpy as np
import tensorflow
import skimage
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Flatten
import keras.models
import keras.utils
import sklearn.model_selection
import sklearn.preprocessing


class CNNModel:
    def __init__(self, random_seed: int, num_classes: int):
        """CNN model wrapper."""

        self.random_seed = random_seed
        self.architecture_id = None

        self.num_classes = np.unique(y).size

        aux = tuple(
            mtd_item for mtd_item in inspect.getmembers(
                self, predicate=inspect.ismethod)
            if mtd_item[0].startswith("architecture_"))

        self._ARCH_NAME, self._ARCH_CALLABLE = zip(*aux)

    def architecture_1(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(Dropout(0.2, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(0.2, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_2(self):
        """Predefined CNN architecture 02."""
        self.model.add(Dense(units=16, activation="relu"))
        self.model.add(Dropout(0.1, seed=self.random_seed))
        self.model.add(Dense(units=32, activation="relu"))
        self.model.add(Dropout(0.1, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def init_architecture(self, architecture_id: int, loss: str,
                          optimizer: str,
                          metrics: t.Union[t.Sequence[str], str]):
        """Init the CNN architecture."""

        def get_architecture_index(architecture_id: int) -> int:
            """Get the correct architecture index in method tuple."""
            return self._ARCH_NAME.index("_".join(("architecture",
                                                   str(architecture_id))))

        np.random.seed(self.random_seed)
        tensorflow.set_random_seed(self.random_seed)

        self.model = keras.models.Sequential()

        self.architecture_id = architecture_id

        self._ARCH_CALLABLE[get_architecture_index(architecture_id)]()

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def validate_architecture(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_splits: int,
            validation_split: float,
            epochs: int,
            batch_size: int,
    ) -> t.Union[t.Sequence[float], float]:
        """Perform an experiment with the selected CNN architecture."""

        kfold = sklearn.model_selection.StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        results = []

        for ind_train, ind_test in kfold.split(X, y):
            X_train, y_train = X[ind_train, :], y[ind_train]
            X_test, y_test = X[ind_test, :], y[ind_test]

            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            y_test = keras.utils.to_categorical(y_test, self.num_classes)

            self.model.fit(
                x=X_train,
                y=y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size)

            res = self.model.evaluate(X_test, y_test)
            results.append(res)

        return np.array(results)

    def freeze_architecture(self) -> None:
        """Save the validated architecture."""
        self.model.save("{}_{}.h5".format("model", str(self.architecture_id)))


def get_data(filepath: str) -> t.Tuple[np.ndarray, np.ndarray]:
    """Return X and y from ``filepath``."""
    X, y = [], []

    classes_dir = os.walk(filepath)
    classes_dir.__next__()

    re_class_name = re.compile(r"(?<=class_)[^\s]+")

    for root, _, files in sorted(classes_dir):
        class_name = re_class_name.search(root).group()
        new_insts = []

        for filename in sorted(files):
            new_insts.append(skimage.io.imread(os.path.join(root, filename)))

        X += new_insts
        y += len(new_insts) * [class_name]

    X, y = np.array(X) // 255, np.array(y)

    return X.reshape(*X.shape, 1), y


if __name__ == "__main__":
    X, y = get_data("./data-augmented-preprocessed")

    y = sklearn.preprocessing.LabelEncoder().fit_transform(y)

    model = CNNModel(random_seed=1234, num_classes=np.unique(y).size)

    results = {}

    for architecture_id in [1]:
        model.init_architecture(
            architecture_id=architecture_id,
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["acc"])

        cur_score = model.validate_architecture(
            X=X,
            y=y,
            n_splits=10,
            validation_split=0.15,
            epochs=12,
            batch_size=32)

        results[architecture_id] = cur_score

    for architecture_id in results:
        cur_results = results[architecture_id]
        print("Architecture {} mean results:".format(architecture_id),
              cur_results.mean(axis=0))

    # model.freeze_architecture()
