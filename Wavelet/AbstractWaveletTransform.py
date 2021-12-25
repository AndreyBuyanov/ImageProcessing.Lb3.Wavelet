from typing import Callable
import numpy as np
from .AbstractWaveletContainer import AbstractWaveletContainer
from .Wavelet import Wavelet
from .Helper import convert_to_gray
from .WaveletHelper import transform


class AbstractWaveletTransform(AbstractWaveletContainer):
    def __init__(self,
                 input_image: np.array):
        super().__init__()
        self.input_image = convert_to_gray(input_image=input_image) \
            if input_image.shape[2] >= 3 \
            else input_image

    def transform(self,
                  levels: int,
                  build_matrix: Callable[[int], np.array]):
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
            transform_h_matrix: np.array = build_matrix(width)
            transform_v_matrix: np.array = build_matrix(height)
            result_image: np.array = transform(
                input_image=work_image,
                transform_h_matrix=transform_h_matrix,
                transform_v_matrix=transform_v_matrix)
            self.wavelets.append(Wavelet(result_image))
            work_image = result_image[0: height // 2, 0: width // 2]
