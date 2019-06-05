"""Module dedicated for image preprocessing steps."""
import typing as t

import matplotlib.pyplot as plt
import skimage.filters
import skimage.color
import skimage.segmentation
import scipy.ndimage
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

        self._opening_mask_1 = (
            scipy.ndimage.morphology.iterate_structure(
                scipy.ndimage.morphology.generate_binary_structure(2, 1),
                iterations=1))

        self._opening_mask_2 = (
            scipy.ndimage.morphology.iterate_structure(
                scipy.ndimage.morphology.generate_binary_structure(2, 2),
                iterations=3))

    def border_crop(self, img: np.ndarray) -> np.ndarray:
        """Crop empty preprocessed image border."""
        coords = np.argwhere(img > 0)

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0) + 1

        return img[x_min:x_max, y_min:y_max]

    def preprocess(self, img: np.ndarray, plot: bool = True) -> np.ndarray:
        """Preprocess the input image.

        The procedures applied to the image are:
            1. RGB to Grayscale
            2. Otsu threshold
            4. Morphologic binary closing
            5. Image empty border crop
        """
        self.img_preprocessed = skimage.color.rgb2gray(img)

        if plot:
            plt.subplot(121)
            plt.imshow(self.img_preprocessed, cmap="gray")

        self.img_preprocessed = skimage.filters.sobel(self.img_preprocessed)

        threshold = skimage.filters.threshold_otsu(self.img_preprocessed)
        self.img_preprocessed = self.img_preprocessed > threshold

        self.img_preprocessed = (
            scipy.ndimage.morphology.binary_closing(
                self.img_preprocessed,
                structure=self._opening_mask_2))

        self.img_preprocessed = self.border_crop(self.img_preprocessed)

        cur_size = np.array(self.img_preprocessed.shape)
        prev_size = 2.0 * cur_size

        while np.any(prev_size > (cur_size * 1.1)):
            for _ in np.arange(2):
                aux = skimage.filters.rank.median(self.img_preprocessed, np.ones((11, 11)))
                self.img_preprocessed[np.logical_and(self.img_preprocessed > 0, aux < 1)] = 0
                self.img_preprocessed = self.border_crop(self.img_preprocessed)

            prev_size = cur_size
            cur_size = np.array(self.img_preprocessed.shape)

        if plot:
            plt.subplot(122)
            plt.imshow(self.img_preprocessed, cmap="gray", vmin=0, vmax=1)
            plt.show()

        return self.img_preprocessed
