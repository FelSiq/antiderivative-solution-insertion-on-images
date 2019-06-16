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
                             output_shape: t.Tuple[int, int],
                             ) -> np.ndarray:
        """."""

        img = skimage.transform.resize(
            image=img,
            order=3,
            anti_aliasing=False,
            output_shape=output_shape)

        img = np.pad(img, pad_width=3, mode="constant", constant_values=0)

        img = img < img.mean()

        return img

    def postprocess(
            self,
            img_base: np.ndarray,
            img_sol: np.ndarray,
            sol_prop_size: t.Tuple[float, float] = (0.15, 0.8),
            sol_prop_local: t.Tuple[np.number, np.number] = (0.80, 0.50),
            ) -> np.ndarray:
        """."""
        img_base = skimage.color.rgb2gray(img_base)

        output_shape = np.ceil(
            sol_prop_size * np.array(img_base.shape)).astype(int)

        img_sol = self._preprocess_solution(
            img=img_sol,
            output_shape=output_shape)

        sol_x, sol_y = (np.array(img_base.shape) * sol_prop_local
                        - np.array(img_sol.shape) // 2).astype(int)

        img_base_slice = img_base[
            sol_x:(sol_x+img_sol.shape[0]+1),
            sol_y:(sol_y+img_sol.shape[1]+1),
        ]

        img_base_slice[np.nonzero(img_sol)] = img_base.min()

        return img_base
