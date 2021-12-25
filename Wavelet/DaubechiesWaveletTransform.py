import numpy as np
from .AbstractWaveletTransform import AbstractWaveletTransform
from .WaveletHelper import build_daubechies_matrix


class DaubechiesWaveletTransform(AbstractWaveletTransform):
    def __init__(self,
                 input_image: np.array):
        super().__init__(
            input_image=input_image)

    def __call__(self,
                 levels: int):
        self.transform(
            levels=levels,
            build_matrix=build_daubechies_matrix)
