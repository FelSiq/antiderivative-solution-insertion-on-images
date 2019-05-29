"""Module dedicated for image preprocessing steps."""
import typing as t

import matplotlib.pyplot as plt
import skimage.filters
import skimage.color
import skimage.segmentation
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

    def preprocess(self, img: np.ndarray, plot: bool = False) -> np.ndarray:
        """Preprocess the input image.

        The procedures applied to the image are:
            1. RGB to Grayscale
            2. Otsu threshold
            3. Image empty border crop
        """
        self.img_preprocessed = skimage.color.rgb2gray(img)

        if plot:
            plt.subplot(121)
            plt.imshow(self.img_preprocessed, cmap="gray")

        self.img_preprocessed = skimage.filters.sobel(self.img_preprocessed)

        threshold = skimage.filters.threshold_otsu(self.img_preprocessed)
        self.img_preprocessed = 255 * (self.img_preprocessed > threshold)

        if plot:
            plt.subplot(122)
            plt.imshow(self.img_preprocessed, cmap="gray")
            plt.show()

        return self.img_preprocessed
