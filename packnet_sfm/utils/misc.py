# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.utils.types import is_list, is_int

########################################################################################################################

def filter_dict(dictionary, keywords):
    """
    Returns only the keywords that are part of a dictionary

    Parameters
    ----------
    dictionary : dict
        Dictionary for filtering
    keywords : list of str
        Keywords that will be filtered

    Returns
    -------
    keywords : list of str
        List containing the keywords that are keys in dictionary
    """
    return [key for key in keywords if key in dictionary]

########################################################################################################################

def make_list(var, n=None):
    """
    Wraps the input into a list, and optionally repeats it to be size n

    Parameters
    ----------
    var : Any
        Variable to be wrapped in a list
    n : int
        How much the wrapped variable will be repeated

    Returns
    -------
    var_list : list
        List generated from var
    """
    var = var if is_list(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var

########################################################################################################################

def same_shape(shape1, shape2):
    """
    Checks if two shapes are the same

    Parameters
    ----------
    shape1 : tuple
        First shape
    shape2 : tuple
        Second shape

    Returns
    -------
    flag : bool
        True if both shapes are the same (same length and dimensions)
    """
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True

########################################################################################################################

def parse_crop_borders(borders, shape):
    """
    Calculate borders for cropping.

    Parameters
    ----------
    borders : tuple
        Border input for parsing. Can be one of the following forms:
        (int, int, int, int): y, height, x, width
        (int, int): y, x --> y, height = image_height - y, x, width = image_width - x
        Negative numbers are taken from image borders, according to the shape argument
        Float numbers for y and x are treated as percentage, according to the shape argument,
            and in this case height and width are centered at that point.
    shape : tuple
        Image shape (image_height, image_width), used to determine negative crop boundaries

    Returns
    -------
    borders : tuple (left, top, right, bottom)
        Parsed borders for cropping
    """
    # Return full image if there are no borders to crop
    if len(borders) == 0:
        return 0, 0, shape[1], shape[0]
    # Copy borders for modification
    borders = list(borders).copy()
    # If borders are 4-dimensional
    if len(borders) == 4:
        borders = [borders[2], borders[0], borders[3], borders[1]]
        if is_int(borders[0]):
            # If horizontal cropping is integer (regular cropping)
            borders[0] += shape[1] if borders[0] < 0 else 0
            borders[2] += shape[1] if borders[2] <= 0 else borders[0]
        else:
            # If horizontal cropping is float (center cropping)
            center_w, half_w = borders[0] * shape[1], borders[2] / 2
            borders[0] = int(center_w - half_w)
            borders[2] = int(center_w + half_w)
        if is_int(borders[1]):
            # If vertical cropping is integer (regular cropping)
            borders[1] += shape[0] if borders[1] < 0 else 0
            borders[3] += shape[0] if borders[3] <= 0 else borders[1]
        else:
            # If vertical cropping is float (center cropping)
            center_h, half_h = borders[1] * shape[0], borders[3] / 2
            borders[1] = int(center_h - half_h)
            borders[3] = int(center_h + half_h)
    # If borders are 2-dimensional
    elif len(borders) == 2:
        borders = [borders[1], borders[0]]
        if is_int(borders[0]):
            # If cropping is integer (regular cropping)
            borders = (max(0, borders[0]),
                       max(0, borders[1]),
                       shape[1] + min(0, borders[0]),
                       shape[0] + min(0, borders[1]))
        else:
            # If cropping is float (center cropping)
            center_w, half_w = borders[0] * shape[1], borders[1] / 2
            center_h, half_h = borders[0] * shape[0], borders[1] / 2
            borders = (int(center_w - half_w), int(center_h - half_h),
                       int(center_w + half_w), int(center_h + half_h))
    # Otherwise, invalid
    else:
        raise NotImplementedError('Crop tuple must have 2 or 4 values.')
    # Assert that borders are valid
    assert 0 <= borders[0] < borders[2] <= shape[1] and \
           0 <= borders[1] < borders[3] <= shape[0], 'Crop borders {} are invalid'.format(borders)
    # Return updated borders
    return borders