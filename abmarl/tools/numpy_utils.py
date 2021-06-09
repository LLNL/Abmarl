import numpy as np


def array_in_array(search_element, elements, axis=0):
    """
    Exact check for a a search element within a collection of elements. Te search element is a
    numpy array, and hte elements is a numpy ndarray. We treat the search element as a single
    entity, so the match has to be across the entire array, not just on the elements within the
    array.

    Args:
        search_element: numpy array that we want to match
        elements: numpy ndarray that we search for a match
        axis: which axis we want to search along. Support 0 and 1.
    """
    if axis == 1:
        elements = elements.transpose()
    for test_element in elements:
        if np.all(search_element == test_element):
            return True
    return False
