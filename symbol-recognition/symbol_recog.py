"""Module dedicated to train the CNN model to object recognition."""
import typing as t
import inspect

import keras.layers
import keras.models
import sklearn.model_selection

import numpy as np


class CNNModel:
    def __init__(self, random_seed: int):
        """CNN model wrapper."""

        self.random_seed = random_seed
        self.architecture_id = None

        aux = tuple(
            mtd_item for mtd_item in inspect.getmembers(
                self, predicate=inspect.ismethod)
            if mtd_item[0].startswith("architecture_")
        )

        self._ARCH_NAME, self._ARCH_CALLABLE = zip(*aux)

    def architecture_1(self):
        """Predefined CNN architecture 01."""
        self.model.add(keras.layers.Conv2D(filters=16, kernel_size=4))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.2, seed=self.random_seed))
        self.model.add(keras.layers.Dense(units=32, activation="relu"))
        self.model.add(keras.layers.Dense(units=18, activation="softmax"))

    def architecture_2(self):
        """Predefined CNN architecture 01."""
        self.model.add(keras.layers.Dense(units=16, activation="relu"))

    def init_architecture(self,
                          architecture_id: int,
                          loss: str,
                          optimizer: str,
                          metrics: t.Union[t.Sequence[str], str]):
        """Init the CNN architecture."""
        def get_architecture_index(architecture_id: int) -> int:
            """Get the correct architecture index in method tuple."""
            return self._ARCH_NAME.index(
                "_".join(("architecture", str(architecture_id))))

        self.model = keras.models.Sequential()

        self.architecture_id = architecture_id

        self._ARCH_CALLABLE[get_architecture_index(architecture_id)]()

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def validate_architecture(
            self,
            X: np.ndarray,
            y: np.ndarray,
            ) -> t.Union[t.Sequence[float], float]:
        """Perform an experiment with the selected CNN architecture."""
        kfold = sklearn.model_selection.StratifiedKFold(
            n_splits=10, shuffle=True, random_state=self.random_state)

        results = sklearn.model_selection.cross_val_score(
            self.model, X, y, cv=kfold)

        return results

    def freeze_architecture(self) -> None:
        """Save the validated architecture."""
        self.model.save("{}_{}.h5".format("model", str(self.architecture_id)))


def get_data(filepath: str) -> t.Tuple[np.ndarray, np.ndarray]:
    """Return X and encoded y from ``filepath``."""
    X, y = None, None

    return X, y


if __name__ == "__main__":
    model = CNNModel(random_seed=1234)

    X, y = get_data()

    results = []

    for architecture_id in range(10):
        model.init_architecture(
            architecture_id=architecture_id,
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=[""])

        cur_score = model.validate_architecture(X=X, y=y)

        results.append((architecture_id, cur_score))

    print(results)

    model.freeze_architecture()
