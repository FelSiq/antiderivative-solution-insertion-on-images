"""Module dedicated to the antiderivative detector."""
import typing as t
import requests
import io

import numpy as np
import wolframalpha
import imageio

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
        self.img_input = None
        self.img_solved = None

        self._preprocessor = preprocess.Preprocessor()
        self._postprocessor = postprocess.Postprocessor()

    def fit(self, img: np.ndarray) -> "Antideriv":
        """Fit an input image into the model."""
        self.img_input = self._preprocessor.preprocess(img.copy())
        return self

    def _get_expression(self) -> str:
        """Get expression from preprocessed input image using CNN."""
        if self.img_input is None:
            raise TypeError("No input image fitted in model.")

        expression = "integrate x^3"

        return expression

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
            img_base=self.img_solved,
            img_sol=img_sol)

        return self.img_solved

    def solve(self, return_text: bool = False) -> np.ndarray:
        """Get solution from preprocessed input image."""
        expression = self._get_expression()
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
