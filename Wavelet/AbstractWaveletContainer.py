from typing import List
from .Wavelet import Wavelet


class AbstractWaveletContainer(object):
    def __init__(self):
        self.wavelets: List[Wavelet] = []

    def __len__(self) -> int:
        return len(self.wavelets)

    def __getitem__(self, item) -> Wavelet:
        return self.wavelets[item]

    def __iter__(self) -> iter:
        return iter(self.wavelets)
