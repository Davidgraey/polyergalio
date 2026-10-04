"""
Utility functions for polyergalio package --- helpers, etc.
"""

import time
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable, Iterable
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from polyergalio.models.constants import EPSILON


def standardize_data(design_matrix: NDArray, axis=0):
    """
    zero tge mean and variance = 1 along axis
    Parameters
    ----------
    design_matrix :
    axis :

    Returns
    -------

    """
    array_mean = np.mean(design_matrix, axis=axis)
    array_std = np.std(design_matrix, axis=axis)

    return (design_matrix - array_mean) / array_std


# -------------- Decorators  --------------
def timed_lru_cache(seconds: int, maxsize: int = 128):
    """from realpython example - uses as @timed_lru_cache"""

    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache


def log_timeit(func: Callable, logger: Optional):
    """
    to use with a created log object
    Parameters
    ----------
    func :
    logger :

    Returns
    -------

    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        message = (
            f"Call of {func.__name__}{args} ({kwargs}) Took {total_time:.4f} seconds"
        )
        if not logger:
            print(message)
        else:
            logger.info(message)

        return result

    return timeit_wrapper


def preformat_expected_shapes(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """conform the shapes of input and targets, and recast into lower-precision dtype"""
    x_prime = np.asarray(x).astype(np.float16)
    y_prime = np.asarray(y).astype(np.float16)

    if x.ndim == 1:  # single sample
        x_prime = x_prime.reshape(1, -1)
    if y.ndim == 1:
        y_prime = y_prime.reshape(-1, 1)

    assert x_prime.shape[-1] == y_prime.shape[-1]

    return x_prime, y_prime


def rolling_windows_nd(data: NDArray,
                       window_size: int,
                       num_overlap: int = 0,
                       axis: int = 0,
                       ) -> NDArray:
    """
    Given a data array, create overlapping, "rolling" windows using numpy stride tricks
    If we provide some data that is (num_samples, sequence, embedding), such as text data
    eg (20, 90, 64)
    and window the 1st axis (sequence) with a window size = 10, and overlap = 2
    We end with the same data, but with expanded dimensions:
    (20, 11, 10, 64), or (num_samples, num_windows, window_size, embedding_size)
    Parameters
    ----------
    data : data array, of at least 2 dimensions
    window : the size (int) of the window
    axis : the target dimension to tile / roll our windows over.  The specified dimension will be expanded into NxM
        dimensions, where Nis the number of windows, and M is the window size.
    num_overlap : the number of indices to overlap each window

    Returns
    -------

    """
    data_shape = data.shape
    target_length = data_shape[axis]

    if num_overlap > window_size:
        print('rollingWindows: num_overlap > window, so setting to window-1')
        num_overlap = window_size - 1 # shift by one

    shift_length = window_size - num_overlap
    num_windows = np.ceil((target_length - window_size + 1) / shift_length).astype(int)

    new_shape = np.insert(
        np.delete(data_shape, axis),
        axis,
        values=[num_windows, window_size]
    )
    # new shape - (batch, window_count, window_size, latent_space)

    # strides = data.strides[:-1] + (data.strides[-1] * num_Shift, data.strides[-1])
    _leading = data.strides[:axis]
    target = [data.strides[axis] * shift_length, data.strides[axis]]  # expanded to 2D
    _trail = data.strides[axis+1:]

    strides = (*_leading, *target, *_trail)

    windowed_indices = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=strides)

    return windowed_indices

def flatten_containers(container: Iterable):
    flattened = []
    for item in container:
        if isinstance(item, list | tuple | set):
            flattened.extend(flatten_containers(item))
        elif isinstance(item, dict):
            flatten_containers(list(item.values()))
        else:  # string, float, int, array
            flattened.append(item)

    return flattened


# -------------- Model Base Class  --------------
def count_elements(value) -> int:
    """Total number of array elements and scalars nested in a dict, list or tuple."""
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(count_elements(member) for member in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count_elements(member) for member in value)
    return int(np.size(value))


# --------------- Standardization / Normalize ---------------
def update_running_standardize(model, new_data_mean, new_data_std, new_data_count) -> None:
    """
    proportionally update the running mean and standard deviation for standardization processes
    Parameters
    ----------
    new_data_mean : mean of the new observations or samples under considerations
    new_data_std : standard deviation of the new observations or samples under considerations
    new_data_count : the number of new samples (for proportionally weighting)

    Returns
    -------
    No returns - we update the self. params with the updated mean of the MEAN and STD DEV
    """
    full_count = model.num_seen_samples + new_data_count
    full_mean = (model.num_seen_samples  * model.x_means + new_data_count * new_data_mean) / full_count
    var1 = model.x_stds ** 2
    var2 = new_data_std ** 2

    # error sum of squares
    sum_square_errors = var1 * (model.num_seen_samples  - 1) + var2 * (new_data_count - 1)
    # total group sum of squares
    sum_squares = (model.x_means - full_mean) ** 2 * model.num_seen_samples  + (new_data_mean - full_mean) ** 2 * new_data_count
    full_var = (sum_square_errors + sum_squares) / (full_count - 1)
    full_std = np.sqrt(full_var)

    model.x_means = full_mean
    model.x_stds = full_std

def standardize(model, data_array: NDArray) -> NDArray:
    """
    Standardize our data array
    Parameters
    ----------
    data_array : numpy array of x-variable

    Returns
    -------
    the mean and standard deviation of the data array
    """
    if model.x_means is None or data_array.shape[-1] != model.x_means.shape[0]:
        log.error("initialize process hasn't been done yet!")

    return (data_array - model.x_means) / (model.x_stds + EPSILON)

def unstandardize(model, data_array: NDArray) -> NDArray:
    """
    unstandardize the data -> convert back into unit space
    Parameters
    ----------
    data_array : numpy array of x-variable

    Returns
    -------
    the data, transformed back into the original unit space
    """
    assert data_array.shape[-1] == model.x_means.shape[0]

    return data_array * (model.x_stds + EPSILON) + model.x_means

def init_standardize(model, x_data: NDArray) -> None:
    """
    initalize the standardize variables for tracking, or update them if
    we're adjusting an already fitted model
    Parameters
    ----------
    x_data : NDArray
    """
    if (model.x_means is None) or (model.num_seen_samples == 0):
        # set up the initial values for the new incoming data
        model.num_seen_samples = x_data.shape[0]
        model.x_means = np.mean(x_data, axis=0)
        model.x_stds = np.std(x_data, axis=0)
    else:
        # update the running standardization parameters with proportional weighting
        model.update_running_standardize(
            new_data_mean=np.mean(x_data, axis=0),
            new_data_std=np.std(x_data, axis=0),
            new_data_count=x_data.shape[0])
    pass
