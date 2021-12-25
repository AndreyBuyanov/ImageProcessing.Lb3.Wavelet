from typing import Callable
import numpy as np
from .Wavelet import Wavelet
from .WaveletHelper import inverse_transform


class AbstractWaveletInverseTransform(object):
    def __init__(self,
                 wavelet: Wavelet):
        self.wavelet = wavelet

    def transform(self,
                  build_matrix: Callable[[int], np.array]) -> np.array:
        work_image = self.wavelet.get()
        height, width = work_image.shape
        transform_h_matrix: np.array = build_matrix(width)
        transform_v_matrix: np.array = build_matrix(height)
        return inverse_transform(
            input_image=work_image,
            transform_h_matrix=transform_h_matrix,
            transform_v_matrix=transform_v_matrix)
