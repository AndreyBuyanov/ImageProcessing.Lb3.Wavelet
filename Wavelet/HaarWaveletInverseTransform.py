import numpy as np
from .AbstractWaveletInverseTransform import AbstractWaveletInverseTransform
from .Wavelet import Wavelet
from .WaveletHelper import build_inverse_haar_matrix


class HaarWaveletInverseTransform(AbstractWaveletInverseTransform):
    def __init__(self,
                 wavelet: Wavelet):
        super().__init__(
            wavelet=wavelet)

    def __call__(self) -> np.array:
        return self.transform(
            build_matrix=build_inverse_haar_matrix)
