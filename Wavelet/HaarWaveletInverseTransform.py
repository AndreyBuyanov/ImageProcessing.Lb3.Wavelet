import numpy as np
from .AbstractWaveletInverseTransform import AbstractWaveletInverseTransform
from .Wavelet import Wavelet
from .WaveletHelper import (
    haar_inverse_h_transform,
    haar_inverse_v_transform
)


class HaarWaveletInverseTransform(AbstractWaveletInverseTransform):
    def __init__(self,
                 wavelet: Wavelet):
        super().__init__(wavelet=wavelet)

    def __call__(self) -> np.array:
        return haar_inverse_h_transform(
            haar_inverse_v_transform(self.wavelet.get()))
