"""Module dedicated to train the CNN model to object recognition."""
import typing as t
import inspect
import os
import re

import numpy as np
import tensorflow
import skimage
from keras.layers import Dropout, Dense, Conv2D, LeakyReLU
from keras.layers import MaxPooling2D, Flatten, BatchNormalization
import keras.models
import keras.utils
import sklearn.model_selection
import sklearn.preprocessing


class Architectures:
    """Various CNN Architectures."""

    def architecture_1(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_2(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_3(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=48, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_4(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=48, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=96, activation="relu"))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_5(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=96, activation="relu"))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_6(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(Conv2D(filters=48, kernel_size=5, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_7(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.1, seed=self.random_seed))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.1, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=96, activation="relu"))
        self.model.add(Dropout(rate=0.1, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_8(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.1, seed=self.random_seed))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.1, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(rate=0.1, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_9(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_10(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_11(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Conv2D(filters=48, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_12(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_13(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))

        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_14(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_15(self):
        """Predefined CNN architecture 01."""
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))
        self.model.add(Conv2D(filters=48, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(rate=0.2, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_16(self):
        """Predefined CNN architecture 16.

        Note: direct variant of the architecture 01.
        """
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_17(self):
        """Predefined CNN architecture 17.

        Note: direct variant of the architecture 06.
        """
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(Conv2D(filters=48, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_18(self):
        """Predefined CNN architecture 18.

        Note: direct variant of the architecture 11.
        """
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Conv2D(filters=48, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_19(self):
        """Predefined CNN architecture 19.

        Note: direct variant of the architecture 12.
        """
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_20(self):
        """Predefined CNN architecture 20.

        Note: direct variant of the architecture 12.
        """
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=32, kernel_size=1, activation="relu"))

        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=64, kernel_size=1, activation="relu"))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_21(self):
        """Predefined CNN architecture 21.

        Note: direct variant of the architecture 14.
        """
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=32, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=32, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(
            Conv2D(filters=64, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=64, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(filters=64, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=64, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_22(self):
        """Predefined CNN architecture 22.

        Note: direct variant of the architecture 14.
        """
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=32, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=32, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(Conv2D(filters=16, kernel_size=1, activation="relu"))

        self.model.add(
            Conv2D(filters=64, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=64, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(filters=64, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=64, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(Conv2D(filters=48, kernel_size=1, activation="relu"))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_23(self):
        """Predefined CNN architecture 23.

        Note: direct variant of the architecture 14.
        """
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=32, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=32, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(Conv2D(filters=16, kernel_size=1, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            Conv2D(filters=64, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=64, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(filters=64, kernel_size=(3, 1), activation="relu"))
        self.model.add(
            Conv2D(filters=64, kernel_size=(1, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))

        self.model.add(Conv2D(filters=48, kernel_size=1, activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.4, seed=self.random_seed))
        self.model.add(Dense(units=self.num_classes, activation="softmax"))

    def architecture_24(self):
        """Predefined CNN architecture 24.

        Note: direct variant of the architecture 16.
        """
        self.model.add(Conv2D(filters=32, kernel_size=5, activation="linear"))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=5, activation="linear"))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="linear"))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        self.model.add(Dense(units=self.num_classes, activation="softmax"))


class CNNModel(Architectures):
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

        self.arch_num = len(self._ARCH_NAME)

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

        keras.backend.clear_session()
        self.model = keras.models.Sequential()

        self.architecture_id = architecture_id

        self._ARCH_CALLABLE[get_architecture_index(architecture_id)]()

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def validate_architecture(
            self,
            X: np.ndarray,
            y: np.ndarray,
            architecture_id: int,
            n_splits: int,
            validation_split: float,
            epochs: int,
            batch_size: int,
            min_delta: float,
            patience: int,
    ) -> t.Union[t.Sequence[float], float]:
        """Perform an experiment with the selected CNN architecture."""

        kfold = sklearn.model_selection.StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        results = []

        es = keras.callbacks.EarlyStopping(
            monitor="val_acc",
            mode="max",
            min_delta=min_delta,
            restore_best_weights=True,
            patience=patience)

        for ind_train, ind_test in kfold.split(X, y):
            X_train, y_train = X[ind_train, :], y[ind_train]
            X_test, y_test = X[ind_test, :], y[ind_test]

            X_train, X_val, y_train, y_val = (
                sklearn.model_selection.train_test_split(
                    X_train,
                    y_train,
                    test_size=validation_split,
                    shuffle=True,
                    stratify=y_train,
                    random_state=self.random_seed))

            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            y_test = keras.utils.to_categorical(y_test, self.num_classes)
            y_val = keras.utils.to_categorical(y_val, self.num_classes)

            model.init_architecture(
                architecture_id=architecture_id,
                loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["acc"])

            self.model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es],
                verbose=1)

            res = self.model.evaluate(X_test, y_test, verbose=1)
            results.append(res)

        return np.array(results)

    def freeze_architecture(self, subpath: t.Optional[str] = None) -> None:
        """Save the validated architecture."""
        prefix = "model"

        if subpath:
            prefix = os.path.join(subpath, prefix)

        if not os.path.exists(subpath):
            os.makedirs(subpath)

        self.model.save("{}_{}.h5".format(prefix, str(self.architecture_id)))


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

    X, y = np.array(X).astype(np.uint8), np.array(y)

    return X.reshape(*X.shape, 1), y


if __name__ == "__main__":
    print("Getting data...")
    X, y = get_data("./data-augmented-preprocessed")
    print("Ok.")

    y = sklearn.preprocessing.LabelEncoder().fit_transform(y)

    model = CNNModel(random_seed=1234, num_classes=np.unique(y).size)

    results = {}

    """
    # Get the performance of all architectures

    print("Testing architectures...")
    for architecture_id in np.arange(1, model.arch_num + 1):
        print("Started training with architecture {}..."
              "".format(architecture_id))

        cur_score = model.validate_architecture(
            X=X,
            y=y,
            architecture_id=architecture_id,
            n_splits=10,
            validation_split=0.1,
            epochs=30,
            batch_size=32,
            patience=8,
            min_delta=0.01)

        results[architecture_id] = cur_score
        print("Architecture {} mean results:".format(architecture_id),
              cur_score.mean(axis=0))

        model.freeze_architecture(subpath="./frozen_models")

    print("Final results")
    for architecture_id in results:
        cur_results = results[architecture_id]
        print("Architecture {} mean results:".format(architecture_id),
              cur_results.mean(axis=0))
    """
    # Get the best model (in this case, architecture #16)
    model.init_architecture(
        architecture_id=16,
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["acc"])

    es = keras.callbacks.EarlyStopping(
        monitor="val_acc",
        mode="max",
        min_delta=0.01,
        restore_best_weights=True,
        patience=5)

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X,
        y,
        test_size=0.1,
        shuffle=True,
        stratify=y,
        random_state=1234)

    y_train = keras.utils.to_categorical(y_train, model.num_classes)
    y_val = keras.utils.to_categorical(y_val, model.num_classes)

    model.model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=35,
        batch_size=64,
        callbacks=[es],
        verbose=1)

    model.freeze_architecture("./frozen_models")

    # Test the best model (architecture 16)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X,
        y,
        test_size=0.1,
        shuffle=True,
        stratify=y,
        random_state=1234)

    y_train = keras.utils.to_categorical(y_train, model.num_classes)
    y_val = keras.utils.to_categorical(y_val, model.num_classes)

    model = keras.models.load_model("./frozen_models/model_16.h5")
    res = model.evaluate(X_train, y_train)
    print(res)
