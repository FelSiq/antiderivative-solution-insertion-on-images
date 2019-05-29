"""Module dedicated for image preprocessing steps."""
import typing as t

import matplotlib.pyplot as plt
import skimage.filters
import skimage.color
import skimage.feature
import numpy as np


class Preprocessor:
    """Call preprocessing methods for antiderivative input images."""

    def __init__(self):
        """

        Parameters
        ----------

        Returns
        ------

        Notes
        -----
        """
        self.img_preprocessed = None  # type: t.Optional[np.ndarray]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess the input image.

        The procedures applied to the image are:
            1. RGB to Grayscale
            2.
            3. Image empty border crop
        """
        self.img_preprocessed = skimage.color.rgb2gray(img)

        """
        threshold = np.percentile(self.img_preprocessed, 0.5)
        threshold = self.img_preprocessed.mean()
        self.img_preprocessed = 255 * (self.img_preprocessed <= threshold)
        """
        self.img_preprocessed = skimage.feature.canny(self.img_preprocessed)

        plt.imshow(self.img_preprocessed, cmap="gray")
        plt.show()

        return self.img_preprocessed
