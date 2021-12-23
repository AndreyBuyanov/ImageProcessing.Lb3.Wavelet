import numpy as np
from .AbstractWaveletInverseTransform import AbstractWaveletInverseTransform
from .Wavelet import Wavelet
from .WaveletHelper import (
    daubechies_inverse_h_transform,
    daubechies_inverse_v_transform
)


class DaubechiesWaveletInverseTransform(AbstractWaveletInverseTransform):
    def __init__(self,
                 wavelet: Wavelet):
        super().__init__(wavelet=wavelet)

    def __call__(self) -> np.array:
        return daubechies_inverse_h_transform(
            daubechies_inverse_v_transform(self.wavelet.get()))
