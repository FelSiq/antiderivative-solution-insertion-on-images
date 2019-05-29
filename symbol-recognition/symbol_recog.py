"""Module dedicated to train the CNN model to object recognition."""
import typing as t

import keras.layers
import keras.models

import numpy as np


class CNNModel:
    def __init__(self):
        """CNN model wrapper."""

        self.init_architecture()

    def init_architecture(self,
                          loss: str,
                          optimizer: str,
                          metrics: t.Union[str, t.Sequence[str]],
                          random_seed: int):
        """Init the CNN architecture."""
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(units=16, activation="relu"))

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # keras.layers.Dense()
        # keras.layers.Activation(activation)
        # keras.layers.Dropout(rate, noise_shape=None, seed=None)
        # keras.engine.input_layer.Input()
        # keras.layers.Flatten(data_format=None)

        # https://keras.io/layers/convolutional/

        # https://keras.io/losses/

        # https://keras.io/optimizers/

        # https://keras.io/layers/pooling/

        # https://keras.io/layers/advanced-activations/

    def validate_architecture(self) -> float:
        """Validate architecture."""

    def freeze_architecture(self) -> None:
        """Save the validated architecture."""


if __name__ == "__main__":
    model = CNNModel()
    # model.freeze_architecture()
