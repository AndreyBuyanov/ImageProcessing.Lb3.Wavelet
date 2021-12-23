import numpy as np
from .Wavelet import Wavelet


class AbstractWaveletInverseTransform(object):
    def __init__(self,
                 wavelet: Wavelet):
        self.wavelet = wavelet

    def __call__(self) -> np.array:
        raise NotImplementedError()
