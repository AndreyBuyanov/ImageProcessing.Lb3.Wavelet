import numpy as np
from .Helper import normalize_image


class Wavelet(object):
    def __init__(self,
                 wavelet: np.array):
        self.wavelet = wavelet

    def get(self) -> np.array:
        return self.wavelet

    def get_normalized(self):
        return normalize_image(self.wavelet)
