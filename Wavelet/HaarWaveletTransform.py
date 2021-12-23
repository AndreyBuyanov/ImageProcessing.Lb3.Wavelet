import numpy as np
from .AbstractWaveletTransform import AbstractWaveletTransform
from .Wavelet import Wavelet
from .WaveletHelper import (
    haar_h_transform,
    haar_v_transform
)


class HaarWaveletTransform(AbstractWaveletTransform):
    def __init__(self,
                 input_image: np.array):
        super().__init__(input_image=input_image)

    def __call__(self,
                 levels: int):
        self.wavelets = []
        work_image: np.array = np.copy(self.input_image)
        for level in range(levels):
            height, width = work_image.shape
            if width <= 4 or height <= 4:
                self.wavelets.append(Wavelet(work_image))
                break
            if width % 2 == 1:
                work_image = np.delete(work_image, width, 1)
            if height % 2 == 1:
                work_image = np.delete(work_image, height, 0)
            result_image: np.array = haar_v_transform(
                haar_h_transform(input_image=work_image))
            self.wavelets.append(Wavelet(result_image))
            work_image = result_image[0: height // 2, 0: width // 2]
