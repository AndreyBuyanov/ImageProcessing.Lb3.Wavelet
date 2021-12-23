import numpy as np
from .AbstractWaveletContainer import AbstractWaveletContainer
from .ImageHelper import convert_to_gray


class AbstractWaveletTransform(AbstractWaveletContainer):
    def __init__(self,
                 input_image: np.array):
        super().__init__()
        self.input_image = convert_to_gray(input_image=input_image) \
            if input_image.shape[2] >= 3 \
            else input_image

    def __call__(self,
                 levels: int):
        raise NotImplementedError()
