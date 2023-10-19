#date: 2023-10-19T17:07:53Z
#url: https://api.github.com/gists/fc0b2ac2622fe0c25d19bd9a35a450e2
#owner: https://api.github.com/users/anshdavid

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import List

import numba as nb
import numpy as np


@nb.njit(nb.types.Tuple((nb.int32[:, :], nb.int32[:]))(nb.int32[:], nb.int32[:], nb.int64, nb.int64))
def numba_product_idx_tup(sizes_array: np.ndarray, current_tuple: np.ndarray, start_idx: int, end_idx: int):
    """
    Iterates through indices from start to end, incrementing tuple counter.

    Parameters
    ----------
    sizes_array : np.ndarray
        array of sizes of components
    current_tuple : np.ndarray
        represents array of array of values for each component (zeros)
    start_idx : int
        index of current tuple to start processing
    end_idx : int
        index of current tuple to end processing

    Returns
    -------
    tuples: np.ndarray
        array with the generated combinations of indexs for each component values
    current_tuple
        represents the last combination generated for the current tuple being processed on
    """

    # initializes a tuples with zeros to store index of computed combination
    tuple_combination_index = np.zeros((end_idx - start_idx, len(sizes_array)), dtype=np.int32)

    # keep track of the current tuple idx
    current_tuple_index = 0

    # stores the current combination
    current_tuple = current_tuple.copy()

    while current_tuple_index < end_idx - start_idx:
        # assign current_tuple to row in the tuples
        tuple_combination_index[current_tuple_index] = current_tuple

        # increments the first element
        current_tuple[0] += 1

        # checks if the first element of current_tuple is equal to the first element of sizes_array.
        # If true, it resets the first element and increments the second element.
        if current_tuple[0] == sizes_array[0]:
            current_tuple[0] = 0
            current_tuple[1] += 1

            # any other elements in current_tuple match corresponding values in sizes_array
            # reset and increment next element
            for i in range(1, len(sizes_array) - 1):
                if current_tuple[i] != sizes_array[i]:
                    break

                # same as before, but in a loop
                current_tuple[i + 1] += 1
                current_tuple[i] = 0

        # next tuple index
        current_tuple_index += 1

    return tuple_combination_index, current_tuple


def chunked_cartesian_product_ids(array_lengths: List[int], chunk_size: int):
    """
    generates chunks of the product of the indices based on array size.

    Parameters
    ----------
    array_lengths : List[int]
        list of integers representing the sizes of the input arrays
    chunk_size : int
        maximum number of combinations to generate in each chunk

    Yields
    ------
    _type_ : Generator
        returns a tuple of generated tuple of product indexs
    """

    # calculate total number of combinations to be generated
    tot_prod = reduce(mul, array_lengths)

    # converts the array_lengths list  array and sort in descending order
    # done to put the largest number at the front
    sizes_array = np.array(array_lengths, dtype=np.int32)
    sorted_idx = np.argsort(sizes_array)[::-1]
    sizes = sizes_array[sorted_idx]

    # initializes current_tuple with zeros
    # to keep track of combination
    current_tuple = np.zeros(len(sizes), dtype=np.int32)
    count_idx = 0

    while True:
        start_idx = count_idx
        end_idx = count_idx + chunk_size

        if end_idx > tot_prod:
            break

        ret_combination_tuples, ret_current_tuple = numba_product_idx_tup(
            sizes, current_tuple, start_idx, end_idx
        )
        current_tuple = ret_current_tuple

        # rearranges the columns to match the original order
        # before yielding
        yield ret_combination_tuples[:, np.argsort(sorted_idx)]

        count_idx += chunk_size

    start_idx = count_idx

    # if chunck was greater than total prod yield entire chunk else yield remaning combinations
    end_idx = count_idx + tot_prod % count_idx if count_idx > 0 else count_idx + tot_prod

    ret_combination_tuples, _ = numba_product_idx_tup(sizes, current_tuple, start_idx, end_idx)

    yield ret_combination_tuples[:, np.argsort(sorted_idx)]


def chunked_cartesian_product(*arrays: np.ndarray, chunk_size: int):
    """
    yeild cartesian production of input arrays

    Parameters
    ----------
    arrays:
        input arrays

    chunk_size : int
        _description_

    Yields
    ------
    _type_
        2D array of combinations
    """

    array_lengths = [len(array) for array in arrays]

    for array_ids_chunk in chunked_cartesian_product_ids(array_lengths, chunk_size):
        # for each array, extract slice using indices
        # contains elements that are part of the combination for the chunk
        slices_lists = [arrays[i][array_ids_chunk[:, i]] for i in range(len(arrays))]

        # stack slices to create a 2D array
        # transpose result, each row represents one combination
        yield np.vstack(slices_lists).swapaxes(0, 1)