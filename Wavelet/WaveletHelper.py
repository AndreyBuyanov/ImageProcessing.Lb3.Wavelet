import numpy as np


def build_haar_matrix(
        size: int) -> np.array:
    result: np.array = np.zeros((size, size))
    for w in range(0, size, 2):
        result[w, w] = 1.0 / np.sqrt(2)
        result[w, w + 1] = 1.0 / np.sqrt(2)
        result[w + 1, w] = -1.0 / np.sqrt(2)
        result[w + 1, w + 1] = 1.0 / np.sqrt(2)
    return result


def build_inverse_haar_matrix(
        size: int) -> np.array:
    return np.transpose(build_haar_matrix(size=size))


def build_daubechies_matrix(
        size: int) -> np.array:
    result: np.array = np.zeros((size, size))
    c1: float = (1.0 + np.sqrt(3.0)) / (4.0 * np.sqrt(2.0))
    c2: float = (3.0 + np.sqrt(3.0)) / (4.0 * np.sqrt(2.0))
    c3: float = (3.0 - np.sqrt(3.0)) / (4.0 * np.sqrt(2.0))
    c4: float = (1.0 - np.sqrt(3.0)) / (4.0 * np.sqrt(2.0))
    for w in range(0, size, 2):
        result[w, w] = c1
        result[w, w + 1] = c2
        result[w, ((w + 2) % size)] = c3
        result[w, ((w + 3) % size)] = c4
        result[w + 1, w] = c4
        result[w + 1, w + 1] = -c3
        result[w + 1, ((w + 2) % size)] = c2
        result[w + 1, ((w + 3) % size)] = -c1
    return result


def build_inverse_daubechies_matrix(
        size: int) -> np.array:
    return np.transpose(build_daubechies_matrix(size=size))


def h_transform(
        input_image: np.array,
        transform_matrix: np.array) -> np.array:
    height, width = input_image.shape
    matrix_row: np.array = transform_matrix
    work_image: np.array = np.zeros(input_image.shape)
    for row in range(height):
        work_image[row] = np.dot(matrix_row, input_image[row])
    approximation = range(0, width, 2)
    difference = range(0, width // 2)
    result: np.array = np.zeros(input_image.shape)
    for row in range(height):
        for i, j in zip(approximation, difference):
            result[row, j] = work_image[row, i]
            result[row, j + width // 2] = work_image[row, i + 1]
    return result


def v_transform(
        input_image: np.array,
        transform_matrix: np.array) -> np.array:
    height, width = input_image.shape
    matrix_col: np.array = transform_matrix
    work_image: np.array = np.zeros(input_image.shape)
    for col in range(width):
        work_image[:, col] = np.dot(matrix_col, input_image[:, col])
    approximation = range(0, height, 2)
    difference = range(0, height // 2)
    result: np.array = np.zeros(input_image.shape)
    for col in range(width):
        for i, j in zip(approximation, difference):
            result[j, col] = work_image[i, col]
            result[j + height // 2, col] = work_image[i + 1, col]
    return result


def inverse_h_transform(
        input_image: np.array,
        transform_matrix: np.array) -> np.array:
    height, width = input_image.shape
    matrix_row: np.array = transform_matrix
    work_image: np.array = np.zeros(input_image.shape)
    for row in range(height):
        collector = range(0, width // 2)
        spreader = range(0, width, 2)
        for i, j in zip(collector, spreader):
            work_image[row, j] = input_image[row, i]
            work_image[row, j + 1] = input_image[row, i + (width // 2)]
    result: np.array = np.zeros(input_image.shape)
    for row in range(height):
        result[row] = np.dot(matrix_row, work_image[row])
    return result


def inverse_v_transform(
        input_image: np.array,
        transform_matrix: np.array) -> np.array:
    height, width = input_image.shape
    matrix_col: np.array = transform_matrix
    work_image: np.array = np.zeros(input_image.shape)
    for col in range(width):
        collector = range(0, height // 2)
        spreader = range(0, height, 2)
        for i, j in zip(collector, spreader):
            work_image[j, col] = input_image[i, col]
            work_image[j + 1, col] = input_image[i + (height // 2), col]
    result: np.array = np.zeros(input_image.shape)
    for col in range(width):
        result[:, col] = np.dot(matrix_col, work_image[:, col])
    return result


def haar_h_transform(
        input_image: np.array) -> np.array:
    _, width = input_image.shape
    transform_matrix: np.array = build_haar_matrix(size=width)
    return h_transform(input_image=input_image, transform_matrix=transform_matrix)


def haar_v_transform(
        input_image: np.array) -> np.array:
    height, _ = input_image.shape
    transform_matrix: np.array = build_haar_matrix(size=height)
    return v_transform(input_image=input_image, transform_matrix=transform_matrix)


def haar_inverse_h_transform(
        input_image: np.array) -> np.array:
    _, width = input_image.shape
    transform_matrix: np.array = build_inverse_haar_matrix(size=width)
    return inverse_h_transform(input_image=input_image, transform_matrix=transform_matrix)


def haar_inverse_v_transform(
        input_image: np.array) -> np.array:
    height, _ = input_image.shape
    transform_matrix: np.array = build_inverse_haar_matrix(size=height)
    return inverse_v_transform(input_image=input_image, transform_matrix=transform_matrix)


def daubechies_h_transform(
        input_image: np.array) -> np.array:
    _, width = input_image.shape
    transform_matrix: np.array = build_daubechies_matrix(size=width)
    return h_transform(input_image=input_image, transform_matrix=transform_matrix)


def daubechies_v_transform(
        input_image: np.array) -> np.array:
    height, _ = input_image.shape
    transform_matrix: np.array = build_daubechies_matrix(size=height)
    return v_transform(input_image=input_image, transform_matrix=transform_matrix)


def daubechies_inverse_h_transform(
        input_image: np.array) -> np.array:
    _, width = input_image.shape
    transform_matrix: np.array = build_inverse_daubechies_matrix(size=width)
    return inverse_h_transform(input_image=input_image, transform_matrix=transform_matrix)


def daubechies_inverse_v_transform(
        input_image: np.array) -> np.array:
    height, _ = input_image.shape
    transform_matrix: np.array = build_inverse_daubechies_matrix(size=height)
    return inverse_v_transform(input_image=input_image, transform_matrix=transform_matrix)
