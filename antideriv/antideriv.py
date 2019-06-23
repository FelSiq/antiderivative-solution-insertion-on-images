"""Module dedicated to the antiderivative detector."""
import typing as t
import io
import re
import os
import requests

import numpy as np
import wolframalpha
import imageio
import skimage.transform
import keras

import preprocess
import postprocess


class Antideriv:
    """Methods for antiderivative detection and symbol recognition."""

    def __init__(self):
        """Main class for antiderivative detection."""
        app_id = 'LHLP7U-HHLKWGU3AT'.lower()

        self._wolfram_client = wolframalpha.Client(app_id)
        self.img_input = None  # type: t.Optional[np.ndarray]
        self.img_solved = None  # type: t.Optional[np.ndarray]
        self.img_segments = None  # type: t.Optional[t.Sequence[np.ndarray]]

        self.models = self._load_models(
            path=os.path.join(
                os.path.realpath(__file__)[:-len(os.path.basename(__file__))],
                "models"))

        self._preprocessor = preprocess.Preprocessor()
        self._postprocessor = postprocess.Postprocessor()

        # Must have correspondence with the class codification
        # used to train the CNN model loaded just above. Don't
        # change the symbol order.
        self._CLASS_SYMBOL = (
            "0",
            "1",
            "x",
            "+",
            "-",
            "/",
            "(",
            ")",
            "e",
            "integrate",
            "d",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        )

        self._RE_FIX_DNOTATION = re.compile(r"(?<=d)\s+(?=.)")

    def _load_models(self, path: str) -> t.Tuple:
        """Load CNN trained models."""
        models_path = (model_name for model_name in os.listdir(path)
                       if model_name.endswith(".h5"))

        loaded_models = (keras.models.load_model(os.path.join(path, model))
                         for model in models_path)

        return tuple(loaded_models)

    def _paint_object(
            self,
            img: np.ndarray,
            start_coord: t.Tuple[int, int],
            window_size: t.Tuple[int, int] = (15, 15),
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

        window_size : :obj:`tuple` with two :obj:`int`, optional
            Size of neighborhood window (in pixels) for each dimension
            to consider while painting the object. The larger the
            ``window_size`` value is, the larger the gaps between parts
            of the same object can be.

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
        if len(window_size) != 2:
            raise ValueError("'window_size' must be a tuple with exactly "
                             "two integer entries.")

        if color is None:
            color = img.max() + 1

        obj_color = img[start_coord]

        if color == obj_color:
            raise ValueError("'color' can't have the same value as the "
                             "given image at the starting coordinates.")

        stack = {start_coord}
        img[start_coord] = color

        while stack:
            cur_coords = stack.pop()

            cur_x, cur_y = cur_coords

            cur_slice = img[(cur_x - window_size[0]):(
                cur_x + window_size[0] + 1), (cur_y - window_size[1]):(
                    cur_y + window_size[1] + 1), ]

            if cur_slice.size:
                slice_coords = np.nonzero(cur_slice == obj_color)
                cur_slice[slice_coords] = color

                cur_x -= window_size[0]
                cur_y -= window_size[1]

                stack.update({(x + cur_x, y + cur_y)
                              for x, y in zip(*slice_coords)})

        # Get currrent object boundary coordinates
        mask = np.argwhere(img == color)
        x_min, y_min = mask.min(axis=0)
        x_max, y_max = mask.max(axis=0)

        return x_min, x_max, y_min, y_max

    def _get_obj_coords(
            self,
            window_size: float = 1.0e-6,
    ) -> t.Tuple[np.ndarray, np.number]:
        """Get coordinates of each object in preprocessed input image.

        Parameters
        ----------
        window_size : :obj:`np.number`, optional
            Size, in proportion to each dimension size of the input image, of
            half the size of the square window to consider as neighborhood for
            each pixel while paiting objects, for object detection.

            For example, if window_size is 0.025, then the window size for
            neighborhood of each pixel will be 5% of the size of each dimension
            of the input image. Note that this parameter must be in the
            interval (0, 1), preferably much smaller than 1.0.

        Returns
        -------
        :obj:`tuple` with :obj:`np.ndarray` and :obj:`np.number`
            * The first entry of the returned tuple is a 2D :obj:`np.ndarray`,
            where each row represents a different object, and each column
            represents the minimum or maximum coordinates of the frame that
            limits each object. More precisely, this array have dimensions Nx4
            where each column represents, in this order, ``x_min``, ``x_max``,
            ``y_min``, ``y_max``.

            * The second entry is the median value of object sizes. Used to
            detect possible remaining noises segmented in the input image.
        """
        if self.img_input is None:
            raise TypeError("'img_input' attribute is None.")

        pad_width = np.ceil(
            np.array(self.img_input.shape) * window_size).astype(int)

        img_painted = np.pad(
            array=self.img_input.copy(),
            pad_width=(
                (pad_width[0], pad_width[0]),
                (pad_width[1], pad_width[1]),
            ),
            mode="constant",
            constant_values=0,
        )

        # The color '0' is the lack of object, and color '1' is a unpainted
        # object.
        color = 2

        obj_coords = []  # type: t.Union[np.ndarray, t.List[np.ndarray]]
        sizes = []  # type: t.List[int]

        for idx_row, idx_col in np.ndindex(self.img_input.shape):
            shifted_coords = (idx_row - pad_width[0], idx_col - pad_width[1])
            if img_painted[shifted_coords] == 1:
                obj_coord = self._paint_object(
                    img=img_painted,
                    start_coord=shifted_coords,
                    window_size=pad_width,
                    color=color)

                obj_coords.append(obj_coord)

                x_min, x_max, y_min, y_max = obj_coord
                sizes.append((1 + x_max - x_min) * (1 + y_max - y_min))

                color += 1

        if not obj_coords:
            raise ValueError("No object detected in input image.")

        obj_coords = np.array(obj_coords)

        # Translate obj_coords to non-padded coordinates
        obj_coords[:, :2] -= pad_width[0]
        obj_coords[:, 2:] -= pad_width[1]

        return obj_coords, np.percentile(sizes, 75)

    def _is_outlier(self, obj: np.ndarray, threshold_val: np.number) -> bool:
        """Check if the given image segment is a possible outlier."""
        return obj.size < (threshold_val * 0.15)

    def _segment_img(self, window_size: np.number = 0.025) -> np.ndarray:
        """Segment the input image into preprocessed units.

        Parameters
        ---------
        window_size : :obj:`np.number`, optional
            Size, in proportion to each dimension size of the input image, of
            half the size of the square window to consider as neighborhood for
            each pixel while paiting objects, for object detection.

            For example, if window_size is 0.025, then the window size for
            neighborhood of each pixel will be 5% of the size of each dimension
            of the input image. Note that this parameter must be in the
            interval (0, 1), preferably much smaller than 1.0.

        Returns
        -------
        :obj:`np.ndarray`
            Numpy array containing all detected objects in preprocessed input
            image.
        """
        if self.img_input is None:
            raise TypeError("'img_input' attribute is None.")

        segments = []  # type: t.Union[t.List[np.ndarray], np.ndarray]

        obj_coords, threshold_val = self._get_obj_coords(
            window_size=window_size)

        for obj_coord in sorted(obj_coords, key=lambda coord: coord[2]):
            x_min, x_max, y_min, y_max = obj_coord

            obj = self.img_input[x_min:(x_max + 1), y_min:(y_max + 1)]

            if not self._is_outlier(obj, threshold_val):
                obj = skimage.transform.rescale(
                    image=obj,
                    scale=45.0 / np.max(obj.shape),
                    anti_aliasing=False,
                    multichannel=False,
                    order=3)

                obj = np.pad(
                    array=obj,
                    pad_width=np.repeat(
                        np.ceil((45 - np.array(obj.shape)) / 2),
                        repeats=2).astype(int).reshape((2, 2)),
                    mode="constant",
                    constant_values=0)[:45, :45]

                obj = (obj >= obj.mean()).astype(np.uint8)

                segments.append(obj)

        segments = np.array(segments)
        segments = segments.reshape(*segments.shape, 1)

        return segments

    def fit(self,
            img: np.ndarray,
            output_file: t.Optional[str] = None,
            window_size: float = 1.0e-6) -> "Antideriv":
        """Fit an input image into the model.

        Parameters
        ----------
        img : :obj:`np.ndarray`
            Input image to fit (and also preprocess).

        output_file : :obj:`str`, optional
            Path of output file to save the preprocessed image.

        window_size : :obj:`np.number`, optional
            Size, in proportion to each dimension size of the input image, of
            half the size of the square window to consider as neighborhood for
            each pixel while paiting objects, for object detection.

            For example, if window_size is 0.025, then the window size for
            neighborhood of each pixel will be 5% of the size of each dimension
            of the input image. Note that this parameter must be in the
            interval (0, 1), preferably much smaller than 1.0.

        Returns
        -------
        Self
        """
        if not isinstance(window_size, (float, int, np.number)):
            raise TypeError("'window_size' must be a number.")

        if not 0.0 < window_size < 1.0:
            raise ValueError("window size must be in (0.0, 1.0) interval, "
                             "preferably much lesser than 1.0.")

        self.img_input = self._preprocessor.preprocess(
            img.copy(), output_file=output_file)

        # Keep original image copy to produce the output later
        self.img_solved = img.copy()

        self.img_segments = self._segment_img(window_size=window_size)

        return self

    def _get_expression(self) -> str:
        """Get expression from preprocessed input image using CNN.

        Returns
        -------
        :obj:`str`
            Text form of the expression in the input image.
        """
        if self.img_segments is None:
            raise TypeError("No input image fitted in model.")

        scores = np.zeros((len(self.img_segments), len(self._CLASS_SYMBOL)))

        for model in self.models:
            preds = model.predict(self.img_segments, verbose=0).argmax(axis=1)

            for seg_idx, pred_idx in enumerate(preds):
                scores[seg_idx, pred_idx] += 1

        expression = [
            self._CLASS_SYMBOL[seg_score]
            for seg_score in scores.argmax(axis=1)
        ]

        # Avoid known common mistakes
        if expression[0] in {"1", "/"}:
            expression[0] = "integrate"

        return self._RE_FIX_DNOTATION.sub("", " ".join(expression))

    def _get_solution(self, expression: str) -> t.Tuple[str, np.ndarray]:
        """Call Wolfram Alpha to get answer to the given ``expression``."""

        def get_solution_image(url: str) -> np.ndarray:
            """Get the image of the solution from Wolfram Alpha."""
            req_ans = requests.get(url)
            return imageio.imread(io.BytesIO(req_ans.content))

        try:
            res = self._wolfram_client.query(expression)

        except Exception as con_err:
            raise ConnectionError("Unable to connect to Wolfram Alpha. "
                                  "Message: {}".format(str(con_err)))

        try:
            ans_plain_text = res["pod"][0]["subpod"]["plaintext"]
            ans_img_url = res["pod"][0]["subpod"]["img"]["@src"]

        except (IndexError, KeyError):
            raise Exception("Wolfram Alpha does not returned an integration "
                            "result. Check your connection and also your "
                            "preprocessed input image.")

        try:
            img_sol = get_solution_image(ans_img_url)

        except Exception as con_err:
            raise ConnectionError("Unable to get the solution image. "
                                  "Message: {}".format(str(con_err)))

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

        ans_plain_text, img_sol = self._get_solution(expression)
        img_ans = self._insert_resolution(img_sol)

        if return_text:
            return img_ans, ans_plain_text

        return img_ans


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("usage", sys.argv[0], "<input image> [output path]")
        exit(1)

    image_path = sys.argv[1]

    input_img = imageio.imread(image_path)

    model = Antideriv().fit(input_img, output_file="../preprocessed.png")
    output_img, ans = model.solve(return_text=True, verbose=True)
    print(ans)

    plt.imshow(output_img, cmap="gray")
    plt.show()

    if len(sys.argv) >= 3:
        plt.imsave(sys.argv[2], output_img, cmap="gray")
        print("Saved output image in '{}' path.".format(sys.argv[2]))
