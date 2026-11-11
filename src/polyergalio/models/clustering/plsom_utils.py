import copy
from collections import deque

import numpy as np
from numpy.typing import NDArray
from polyergalio.models.constants import EPSILON

# from smlib.mltools import log


def free_energy(scores: NDArray, temperature: float, axis: int = -1) -> NDArray:
    """
    Soft-minimum of scores, F = -T log sum_k exp(-score_k / T)

    Parameters
    ----------
    scores : distances or neighborhood-smoothed distances, lowest wins
    temperature : softness of the minimum; 0 or below gives the hard minimum
    axis : axis holding the candidates (prototypes)

    Returns
    -------
    free energy with `axis` reduced away
    """
    if temperature <= 0:
        return scores.min(axis=axis)

    logits = -scores / temperature
    peak = logits.max(axis=axis, keepdims=True)
    total = np.exp(logits - peak).sum(axis=axis)
    return -temperature * (np.squeeze(peak, axis=axis) + np.log(total))


def umat_fill_outer(u_matrix, cols, rows):
    """
    takes a partially constructed Rectangular U_matrix or distance matrix and fills in missing values

    """
    umatrix = copy.copy(u_matrix)
    for row_i in range(rows):
        for col_i in range(cols):
            # let's resolve corner cases, then do edges
            if row_i == 0 and col_i == 0:
                # topleft - only right and lower
                lower_v = umatrix[row_i + 1, col_i]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((right_v, lower_v))
            elif row_i == 0 and col_i == cols - 1:
                # topright - only right and lower
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                umatrix[row_i, col_i] = np.mean((left_v, lower_v))
            elif row_i == rows - 1 and col_i == 0:
                # bottomleft - only right and upper
                right_v = umatrix[row_i, col_i + 1]
                upper_v = umatrix[row_i - 1, col_i]
                umatrix[row_i, col_i] = np.mean((right_v, upper_v))
            elif row_i == rows - 1 and col_i == cols - 1:
                # bottomright - only left and upper
                upper_v = umatrix[row_i - 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                umatrix[row_i, col_i] = np.mean((left_v, upper_v))

            # Left Edge
            elif 0 < row_i < rows - 1 and row_i % 2 == 0 and col_i == 0:
                upper_v = umatrix[row_i - 1, col_i]
                lower_v = umatrix[row_i + 1, col_i]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((right_v, lower_v, upper_v))

            # Right Edge
            elif 0 < row_i < rows - 1 and row_i % 2 == 0 and col_i == cols - 1:
                upper_v = umatrix[row_i - 1, col_i]
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                umatrix[row_i, col_i] = np.mean((left_v, lower_v, upper_v))

            # Top Edge - corners calculated
            elif row_i == 0 and 0 < col_i < cols - 1 and col_i % 2 == 0:
                # print('top_edge')
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((left_v, lower_v, right_v))

            # Bottom Edge - corners calculated
            elif row_i == rows - 1 and 0 < col_i < cols - 1 and col_i % 2 == 0:
                # print('bottom_edge')
                upper_v = umatrix[row_i - 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((left_v, upper_v, right_v))

    return umatrix


def umat_infill(u_matrix, cols, rows):
    """
    Fills in the 'inbetween' values that are not directly calculable.
    This is highly inelegant
    """
    umatrix = copy.copy(u_matrix)
    for row_i in range(rows):
        for col_i in range(cols):
            # evens
            if 0 < row_i < rows - 1 and row_i % 2 == 0 and 0 < col_i < cols - 1 and col_i % 2 == 0:
                upper_v = umatrix[row_i - 1, col_i]
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((upper_v, lower_v, left_v, right_v))

    # fill odds
    for row_i in range(rows):
        for col_i in range(cols):
            if 0 < row_i < rows - 1 and row_i % 2 != 0 and 0 < col_i < cols - 1 and col_i % 2 != 0:
                upper_v = umatrix[row_i - 1, col_i]
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((upper_v, lower_v, left_v, right_v))

    return umatrix


class Watershed(object):
    MASK = -2
    WSHD = 0
    INIT = -1
    INQE = -3

    def __init__(self, levels=25):
        self.levels = levels

    # Neighbour (coordinates of) pixels, including the given pixel.
    def _get_neighbors(self, height, width, pixel):
        return np.mgrid[
            max(0, pixel[0] - 1):min(height, pixel[0] + 2),
            max(0, pixel[1] - 1):min(width, pixel[1] + 2)
        ].reshape(2, -1).T

    def apply(self, image):
        current_label = 0
        flag = False
        fifo = deque()

        height, width = image.shape
        total = height * width
        labels = np.full((height, width), self.INIT, np.int32)

        reshaped_image = image.reshape(total)
        # [y, x] pairs of pixel coordinates of the flattened image.
        pixels = np.mgrid[0:height, 0:width].reshape(2, -1).T
        # Coordinates of neighbour pixels for each pixel.
        neighbours = np.array([self._get_neighbors(height, width, p) for p in pixels])
        if len(neighbours.shape) == 3:
            # Case where all pixels have the same number of neighbours.
            neighbours = neighbours.reshape(height, width, -1, 2)

        else:
            # Case where pixels may have a different number of pixels.
            neighbours = neighbours.reshape(height, width)

        indices = np.argsort(reshaped_image)
        sorted_image = reshaped_image[indices]
        sorted_pixels = pixels[indices]

        # self.levels evenly spaced steps from minimum to maximum.
        levels = np.linspace(sorted_image[0], sorted_image[-1], self.levels)
        level_indices = []
        current_level = 0

        # Get the indices that deleimit pixels with different values.
        for i in range(total):
            if sorted_image[i] > levels[current_level]:
                # Skip levels until the next highest one is reached.
                while sorted_image[i] > levels[current_level]: current_level += 1
                level_indices.append(i)
        level_indices.append(total)

        start_index = 0
        for stop_index in level_indices:
            # Mask all pixels at the current level.
            for p in sorted_pixels[start_index:stop_index]:
                labels[p[0], p[1]] = self.MASK
                # Initialize queue with neighbours of existing basins at the current level.
                for q in neighbours[p[0], p[1]]:
                    # p == q is ignored here because labels[p] < WSHD
                    if labels[q[0], q[1]] >= self.WSHD:
                        labels[p[0], p[1]] = self.INQE
                        fifo.append(p)
                        break

            # Extend basins.
            while fifo:
                p = fifo.popleft()
                # Label p by inspecting neighbours.
                for q in neighbours[p[0], p[1]]:
                    # Don't set lab_p in the outer loop because it may change.
                    lab_p = labels[p[0], p[1]]
                    lab_q = labels[q[0], q[1]]
                    if lab_q > 0:
                        if lab_p == self.INQE or (lab_p == self.WSHD and flag):
                            labels[p[0], p[1]] = lab_q
                        elif lab_p > 0 and lab_p != lab_q:
                            labels[p[0], p[1]] = self.WSHD
                            flag = False
                    elif lab_q == self.WSHD:
                        if lab_p == self.INQE:
                            labels[p[0], p[1]] = self.WSHD
                            flag = True
                    elif lab_q == self.MASK:
                        labels[q[0], q[1]] = self.INQE
                        fifo.append(q)

            # Detect and process new minima at the current level.
            for p in sorted_pixels[start_index:stop_index]:
                # p is inside a new minimum. Create a new label.
                if labels[p[0], p[1]] == self.MASK:
                    current_label += 1
                    fifo.append(p)
                    labels[p[0], p[1]] = current_label
                    while fifo:
                        q = fifo.popleft()
                        for r in neighbours[q[0], q[1]]:
                            if labels[r[0], r[1]] == self.MASK:
                                fifo.append(r)
                                labels[r[0], r[1]] = current_label

            start_index = stop_index

        return labels


def blur(image: NDArray) -> NDArray:
    """
    blur the image array by the given kernel
    :param image: np array - imagelike
    :return: convolved kernel blur image
    """
    image_array = copy.deepcopy(image)
    kernel = np.array([1.0, 2.0, 1.0])
    image_array = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, image_array)
    image_array = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, image_array)
    return image_array
