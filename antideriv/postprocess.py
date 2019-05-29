"""Module dedicated for postprocessing the input image."""
import typing as t

import numpy as np


class Postprocessor:
    """Class with various methods for postprocessing the input image."""
    def __init__(self):
        """
        Parameters
        ----------

        Returns
        -------

        Note
        ----
        """
        self.img_postprocessed = None  # type: t.Optional[np.ndarray]

    def postprocess(self,
                    img_base: np.ndarray,
                    img_sol: np.ndarray) -> np.ndarray:
        """."""

        return img_base
