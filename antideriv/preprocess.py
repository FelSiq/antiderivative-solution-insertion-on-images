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
        """Preprocess the input image of Antideriv class.

        Notes
        -----
        Check ``preprocess`` method documentation for more information.
        """
        self.img_preprocessed = None  # type: t.Optional[np.ndarray]

        self._opening_mask_1 = (scipy.ndimage.morphology.iterate_structure(
            scipy.ndimage.morphology.generate_binary_structure(2, 1),
            iterations=1))

        self._opening_mask_2 = (scipy.ndimage.morphology.iterate_structure(
            scipy.ndimage.morphology.generate_binary_structure(2, 2),
            iterations=3))

    @classmethod
    def border_crop(cls, img: np.ndarray) -> np.ndarray:
        """Crop empty preprocessed image border."""
        coords = np.argwhere(img > 0)

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0) + 1

        return img[x_min:x_max, y_min:y_max]

    def preprocess(self,
                   img: np.ndarray,
                   plot: bool = False,
                   output_file: t.Optional[str] = None) -> np.ndarray:
        """Preprocess the input image.

        The procedures applied to the image are:
            1. RGB to Grayscale
            2. Sobel filter
            3. Otsu threshold
            4. Morphologic binary closing
            5. Image empty border crop
            6. (Repeat until image size does not change significantly)
                6.1. Rank 11 median filter, but only removing non-zero values,
                    i.e., it does not affect pixels with zero value.
                6.2. Empty border crop

        Arguments
        ---------
        img : :obj:`np.ndarray`
            Image to be processed.

        plot : :obj:`bool`, optional
            If True, plot the final preprocessed image.

        output_file : :obj:`str`, optional
            If not :obj:`NoneType`, save the final preprocessed image in the
            given filepath.

        Returns
        -------
        :obj:`np.ndarray`
            Preprocessed image.
        """
        self.img_preprocessed = skimage.color.rgb2gray(img)

        if plot:
            plt.subplot(121)
            plt.imshow(self.img_preprocessed, cmap="gray")

        self.img_preprocessed = skimage.filters.sobel(self.img_preprocessed)

        threshold = skimage.filters.threshold_otsu(self.img_preprocessed)
        self.img_preprocessed = self.img_preprocessed > threshold

        self.img_preprocessed = (scipy.ndimage.morphology.binary_closing(
            self.img_preprocessed, structure=self._opening_mask_2))

        self.img_preprocessed = Preprocessor.border_crop(self.img_preprocessed)

        cur_size = np.array(self.img_preprocessed.shape)
        prev_size = 2.0 * cur_size

        while np.any(prev_size > (cur_size * 1.1)):
            for _ in np.arange(2):
                aux = skimage.filters.rank.median(self.img_preprocessed,
                                                  np.ones((11, 11)))
                self.img_preprocessed[np.logical_and(self.img_preprocessed > 0,
                                                     aux < 1)] = 0
                self.img_preprocessed = Preprocessor.border_crop(
                    self.img_preprocessed)

            prev_size = cur_size
            cur_size = np.array(self.img_preprocessed.shape)

        if plot:
            plt.subplot(122)
            plt.imshow(self.img_preprocessed, cmap="gray", vmin=0, vmax=1)
            plt.show()

        self.img_preprocessed = self.img_preprocessed.astype(np.uint8)

        if output_file:
            plt.imsave(output_file, 255 * self.img_preprocessed, cmap="gray")

        return self.img_preprocessed
