"""Module dedicated for postprocessing the input image."""
import typing as t

import numpy as np
import skimage.transform
import skimage.color


class Postprocessor:
    """Class with various methods for postprocessing the input image."""
    def __init__(self):
        """."""
        self.img_postprocessed = None  # type: t.Optional[np.ndarray]

        self._dilation_mask = np.ones((3, 3))

    def _preprocess_solution(self,
                             img: np.ndarray,
                             scale: float,
                             ) -> np.ndarray:
        """Preprocess the solution image given by Wolfram Alpha."""

        img = skimage.transform.rescale(
            image=img,
            scale=scale,
            order=3,
            anti_aliasing=False,
            multichannel=False)

        img = np.pad(img, pad_width=3, mode="constant", constant_values=0)

        img = img < img.mean()

        return img

    def postprocess(
            self,
            img_base: np.ndarray,
            img_sol: np.ndarray,
            sol_prop_size: float = 0.20,
            sol_prop_local: t.Tuple[np.number, np.number] = (0.80, 0.50),
            ) -> np.ndarray:
        """Process the solution image and insert it to the ``img_base``.

        Arguments
        ---------
        img_base : :obj:`np.ndarray`
            Image to serve as base, i.e., the target of the insertion.

        img_sol : :obj:`np.ndarray`
            Image of the integration solution of the input image.

        sol_prop_size : :obj:`float`, optional
            Proportion of space which the ``img_sol`` must take from
            the ``img_base`` based on axis 0 (rows).

        sol_prop_local : :obj:`tuple` with two :obj:`float`, optional
            Tuple containing the proportion of ``img_base`` where the
            center of the postprocessed ``img_sol`` must be.

        Returns
        -------
        :obj:`np.ndarray`
            ``img_base`` with postprocessed ``img_sol`` inserted inside it.
        """
        img_base = skimage.color.rgb2gray(img_base)

        img_sol = self._preprocess_solution(
            img=img_sol,
            scale=(img_base.shape[0] * sol_prop_size) / img_sol.shape[0])

        sol_x, sol_y = (np.array(img_base.shape) * sol_prop_local
                        - np.array(img_sol.shape) // 2).astype(int)

        img_base_slice = img_base[
            sol_x:(sol_x+img_sol.shape[0]+1),
            sol_y:(sol_y+img_sol.shape[1]+1),
        ]

        img_base_slice[np.nonzero(img_sol)] = img_base.min()

        return img_base
