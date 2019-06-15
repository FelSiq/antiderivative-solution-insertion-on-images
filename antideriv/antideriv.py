"""Module dedicated to the antiderivative detector."""
import typing as t
import requests
import io

import numpy as np
import wolframalpha
import imageio
import skimage.transform
import keras

import preprocess
import postprocess

# import matplotlib.pyplot as plt


class Antideriv:
    """Methods for antiderivative detection and symbol recognition."""

    def __init__(self):
        """Main class for antiderivative detection.

        Parameters
        ----------

        Returns
        -------

        Notes
        -----

        """
        app_id = 'LHLP7U-HHLKWGU3AT'.lower()

        self._wolfram_client = wolframalpha.Client(app_id)
        self.img_input = None  # type: t.Optional[np.ndarray]
        self.img_solved = None  # type: t.Optional[np.ndarray]
        self.img_segments = None  # type: t.Optional[t.Sequence[np.ndarray]]

        self.model = keras.models.load_model("model_16.h5")

        self._preprocessor = preprocess.Preprocessor()
        self._postprocessor = postprocess.Postprocessor()

        # Must have correspondence with the class codification
        # used to train the CNN model loaded just above. Don't
        # change the symbol order.
        self._CLASS_SYMBOL = (
            "0",
            "9",
            "x",
            "e",
            "+",
            "-",
            "(",
            ")",
            "/",
            "integrate",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
        )

    def _paint_object(
            self,
            img: np.ndarray,
            start_coord: t.Tuple[int, int],
            window_size: int = 3,
            color: t.Optional[int] = None,
    ) -> t.Tuple[int, int, int, int]:
        """Paint object under the ``star_coord`` in the input image.

        The paiting is in-place.

        Arguments
        ---------
        img : :obj:`np.ndarray`
            Image to paint in-place.

        start_coord : :obj:`tuple` with two :obj:`int`
            A tuple containing the starting coordinates, x and y, of
            some previously segmented object.

        window_size : :obj:`int`, optional
            Size of neighborhood window (in pixels) to consider while
            paiting the object. The larger the window_size is, the
            larger the gaps between parts of the same object can be.

        color : :obj:`int`, optional
            Color to paint the detected object. The color can't be the
            same color as the object (i.e., must be different from the
            color at ``start_coord`` coordinates. If :obj:`NoneType`,
            then the default color will be max(img) + 1.

        Returns
        -------
        :obj:`tuple` with four :obj:`int`
            Tuple containing four integers relative to, respectively,
            ``x_min``, ``x_max``, ``y_min`` and ``y_max``.
        """
        if color is None:
            color = img.max() + 1

        obj_color = img[start_coord]

        if color == obj_color:
            raise ValueError("'color' can't have the same value as the "
                             "given image at the starting coordinates.")

        stack = [start_coord]
        img[start_coord] = color

        while stack:
            cur_coords = stack.pop()

            cur_x, cur_y = cur_coords

            cur_slice = img[(cur_x - window_size):(cur_x + window_size + 1), (
                cur_y - window_size):(cur_y + window_size + 1)]

            if cur_slice.size:
                slice_coords = np.nonzero(cur_slice == obj_color)
                cur_slice[slice_coords] = color

                stack += [[x + cur_x - window_size, y + cur_y - window_size]
                          for x, y in zip(*slice_coords)]

        mask = np.argwhere(img == color)
        x_min, y_min = mask.min(axis=0)
        x_max, y_max = mask.max(axis=0)

        return x_min, x_max, y_min, y_max

    def _get_size_threshold(self, sizes: np.ndarray, whis: np.number = 1.50
                            ) -> t.Tuple[np.number, np.number]:
        """."""
        _q25, _q75, threshold_val = np.percentile(sizes, (25, 75, 50))
        iqr_whis = whis * (_q75 - _q25)
        return threshold_val, iqr_whis

    def _get_obj_coords(
            self,
            whis: np.number = 1.50,
            window_size: int = 3,
    ) -> t.Tuple[np.ndarray, t.Tuple[np.number, np.number]]:
        """."""
        if self.img_input is None:
            raise TypeError("'img_input' attribute is None.")

        obj_coords = []
        sizes = []

        img_painted = np.pad(
            array=self.img_input.copy(),
            pad_width=window_size,
            mode="constant",
            constant_values=0,
        )

        # The color '0' is the lack of object, and color '1' is a unpainted
        # object.
        color = 2

        for i, j in np.ndindex(self.img_input.shape):
            shifted_coords = (i + window_size, j + window_size)
            if img_painted[shifted_coords] == 1:
                obj_coord = self._paint_object(
                    img=img_painted,
                    start_coord=shifted_coords,
                    window_size=window_size,
                    color=color)

                obj_coords.append(np.array(obj_coord) - window_size)

                x_min, x_max, y_min, y_max = obj_coord
                sizes.append((1 + x_max - x_min) * (1 + y_max - y_min))

                color += 1

        threshold_info = self._get_size_threshold(np.array(sizes), whis=whis)

        return np.array(obj_coords), threshold_info

    def _segment_img(self, whis: np.number = 1.50,
                     window_size: int = 3) -> t.Sequence[np.ndarray]:
        """Segment the input image into preprocessed units."""
        if self.img_input is None:
            raise TypeError("'img_input' attribute is None.")

        segments = []  # type: t.Union[t.List[np.ndarray], np.ndarray]

        obj_coords, threshold_info = self._get_obj_coords(
            whis=whis, window_size=window_size)

        threshold_val, iqr_whis = threshold_info

        for obj_coord in sorted(obj_coords, key=lambda coord: coord[2]):
            x_min, x_max, y_min, y_max = obj_coord

            obj = self.img_input[x_min:(x_max + 1), y_min:(y_max + 1)]

            if np.abs(obj.size - threshold_val) <= iqr_whis:
                obj = skimage.transform.resize(
                    image=obj,
                    output_shape=(32, 32),
                    anti_aliasing=False,
                    order=3)

                obj = (obj >= obj.mean()).astype(np.uint8)

                segments.append(obj)

        segments = np.array(segments)
        segments = segments.reshape(*segments.shape, 1)

        return segments

    def fit(self,
            img: np.ndarray,
            output_file: t.Optional[str] = None,
            whis: np.number = 1.50,
            window_size: int = 10) -> "Antideriv":
        """Fit an input image into the model.

        Parameters
        ----------
        img : :obj:`np.ndarray`
            Input image to fit (and also preprocess).

        output_file : :obj:`str`, optional
            Path of output file to save the preprocessed image.

        Returns
        -------
        Self.
        """
        self.img_input = self._preprocessor.preprocess(
            img.copy(), output_file=output_file)

        # Keep original image copy to produce the output later
        self.img_solved = img.copy()

        self.img_segments = self._segment_img(
            whis=whis, window_size=window_size)

        return self

    def _get_expression(self) -> str:
        """Get expression from preprocessed input image using CNN."""
        if self.img_segments is None:
            raise TypeError("No input image fitted in model.")
        """
        np.set_printoptions(threshold=np.inf)
        for seg in self.img_segments:
            aux = seg.reshape((32, 32))
            print(aux)
            plt.imshow(aux.reshape((32, 32)))
            plt.show()
        """

        preds = self.model.predict(self.img_segments, verbose=0)

        expression = [
            self._CLASS_SYMBOL[symbol_ind]
            for symbol_ind in preds.argmax(axis=1)
        ]

        return " ".join(expression)

    def _get_solution(self, expression: str) -> t.Tuple[str, np.ndarray]:
        """Call Wolfram Alpha to get answer to the given ``expression``."""

        def get_solution_image(url: str) -> np.ndarray:
            """Get the image of the solution from Wolfram Alpha."""
            req_ans = requests.get(ans_img_url)
            return imageio.imread(io.BytesIO(req_ans.content))

        res = self._wolfram_client.query(expression)
        ans_plain_text = res["pod"][0]["subpod"]["plaintext"]
        ans_img_url = res["pod"][0]["subpod"]["img"]["@src"]

        img_sol = get_solution_image(ans_img_url)

        return ans_plain_text, img_sol

    def _insert_resolution(self, img_sol: np.ndarray) -> np.ndarray:
        """Insert resolution in input image."""
        self.img_solved = self._postprocessor.postprocess(
            img_base=self.img_solved, img_sol=img_sol)

        return self.img_solved

    def solve(self, return_text: bool = False, verbose: bool = True
              ) -> t.Union[t.Tuple[np.ndarray, str], np.ndarray]:
        """Get solution from preprocessed input image.

        Arguments
        ---------
        return_text : :obj:`bool`, optional
            If True, also return the solution as plain text.

        verbose : :obj:`bool`, optional
            If True, enable print messages produced along the
            solving process.

        Returns
        -------
        If ``return_text`` is False:
            :obj:`np.ndarray`
                Input image with solution inserted.
        else:
            :obj:`tuple` with :obj:`np.ndarray` and :obj:`str`
                Input image with solution inserted and the
                solution itself as plain text.
        """
        expression = self._get_expression()

        if verbose:
            print("Expression: {}".format(expression))

        exit(1)

        ans_plain_text, img_sol = self._get_solution(expression)
        img_ans = self._insert_resolution(img_sol)

        if return_text:
            return img_ans, ans_plain_text

        return img_ans


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        print("usage", sys.argv[0], "<input image>")
        exit(1)

    image_path = sys.argv[1]

    input_img = imageio.imread(image_path)

    model = Antideriv().fit(input_img, output_file="../preprocessed.jpg")
    img, ans = model.solve(return_text=True, verbose=True)
    print(ans)
