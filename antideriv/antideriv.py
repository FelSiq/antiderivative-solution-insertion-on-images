"""Module dedicated to the antiderivative detector."""
import typing as t
import requests
import io

import numpy as np
import wolframalpha
import imageio
import keras

import preprocess
import postprocess


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

    def _segment_img(self) -> t.Sequence[np.ndarray]:
        """Segment the input image into preprocessed units."""
        segments = []

        aux = "../symbol-recognition/data-augmented-preprocessed/class_"
        segments.append(imageio.imread("".join((aux, "18/18_40.png"))))
        segments.append(imageio.imread("".join((aux, "3/3_40.png"))))
        segments.append(imageio.imread("".join((aux, "11/11_40.png"))))
        segments.append(imageio.imread("".join((aux, "13/13_40.png"))))
        segments.append(imageio.imread("".join((aux, "7/7_40.png"))))
        segments.append(imageio.imread("".join((aux, "11/11_40.png"))))
        """
            Necessary steps:
            - Cut image into separated pieces
            - Resize to 32x32 images
        """

        segments = np.array(segments)
        segments = segments.reshape(*segments.shape, 1) // 255

        return segments

    def fit(self, img: np.ndarray) -> "Antideriv":
        """Fit an input image into the model."""
        self.img_input = self._preprocessor.preprocess(img.copy())
        self.img_segments = self._segment_img()

        return self

    def _get_expression(self) -> str:
        """Get expression from preprocessed input image using CNN."""
        if self.img_segments is None:
            raise TypeError("No input image fitted in model.")

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

    def solve(self, return_text: bool = False,
              verbose: bool = True) -> np.ndarray:
        """Get solution from preprocessed input image."""
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

    if len(sys.argv) <= 1:
        print("usage", sys.argv[0], "<input image>")
        exit(1)

    image_path = sys.argv[1]

    input_img = imageio.imread(image_path)

    model = Antideriv().fit(input_img)
    img, ans = model.solve(return_text=True)
    print(ans)
