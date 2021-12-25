import numpy as np


def normalize_image(
        input_image: np.array) -> np.array:
    result_image: np.array = np.zeros(input_image.shape)
    input_max = input_image.max()
    input_min = input_image.min()
    input_range = input_max - input_min
    height, width = input_image.shape
    for y in range(height):
        for x in range(width):
            input_value = input_image[y][x]
            scaled_input_value = (input_value - input_min) / input_range if input_range != 0 else 0
            result_image[y][x] = scaled_input_value * 255.0
    return result_image


def convert_to_gray(
        input_image: np.array) -> np.array:
    height, width, depth = input_image.shape
    result_image: np.array = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            result_image[y][x] \
                = input_image[y][x][0] * 0.299 \
                + input_image[y][x][1] * 0.587 \
                + input_image[y][x][2] * 0.114
    return result_image


def calculate_psnr(
        img1: np.array,
        img2: np.array,
        max_value=255):
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))
