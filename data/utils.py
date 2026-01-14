import numpy as np


def array_pad_shrink(
    array: np.array,
    output_length: int,
    padding_value: float,
) -> np.array:
    """
    If array is longer then output_length, then it gets shortened to output_length, otherwise it
    gets padded with padding_value.
    :param array: 1D Numpy Array
    :param output_length: length of the output array
    :param padding_value: value used to pad the array
    :return: padded/shrunk array
    """
    assert output_length > 0
    return np.pad(
        array, (0, max(0, output_length - len(array))), constant_values=padding_value
    )[:output_length]
