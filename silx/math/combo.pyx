# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""This module provides combination of statistics as single operation.

It contains:

- :func:`min_max` that computes min/max (and optionally positive min)
  and indices of first occurences (i.e., argmin/argmax) in a single pass.
- :func:`mean_std` that computes mean and std in a single pass.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "20/12/2016"

cimport cython
from libc.math cimport isnan, sqrt, NAN

import numpy


ctypedef fused _number:
    float
    double
    signed char
    signed short
    signed int
    signed long
    unsigned char
    unsigned short
    unsigned int
    unsigned long long


### Min Max Positive Min combo ###

class _MinMaxResult(object):
    """Object storing result from :func:`min_max`"""

    def __init__(self, minimum, min_pos, maximum,
                 argmin, argmin_pos, argmax):
        self._minimum = minimum
        self._min_positive = min_pos
        self._maximum = maximum

        self._argmin = argmin
        self._argmin_positive = argmin_pos
        self._argmax = argmax

    minimum = property(
        lambda self: self._minimum,
        doc="Minimum value of the array")
    maximum = property(
        lambda self: self._maximum,
        doc="Maximum value of the array")

    argmin = property(
        lambda self: self._argmin,
        doc="Index of the first occurence of the minimum value")
    argmax = property(
        lambda self: self._argmax,
        doc="Index of the first occurence of the maximum value")

    min_positive = property(
        lambda self: self._min_positive,
        doc="""Strictly positive minimum value
        
        It is None if no value is strictly positive.
        """)
    argmin_positive = property(
        lambda self: self._argmin_positive,
        doc="""Index of the strictly positive minimum value.

        It is None if no value is strictly positive.
        It is the index of the first occurence.""")

    def __getitem__(self, key):
        if key == 0:
            return self.minimum
        elif key == 1:
            return self.maximum
        else:
            raise IndexError("Index out of range")


@cython.boundscheck(False)
@cython.wraparound(False)
def _min_max(_number[:] data, bint min_positive=False):
    """See :func:`min_max` for documentation."""
    cdef:
        _number value, minimum, minpos, maximum
        unsigned int length
        unsigned int index = 0
        unsigned int min_index = 0
        unsigned int min_pos_index = 0
        unsigned int max_index = 0

    length = len(data)

    if length == 0:
        raise ValueError('Zero-size array')

    with nogil:
        # Init starting values
        value = data[0]
        minimum = value
        maximum = value
        if min_positive and value > 0:
            min_pos = value
        else:
            min_pos = 0

        if _number in cython.floating:
            # For floating, loop until first not NaN value
            for index in range(length):
                value = data[index]
                if not isnan(value):
                    minimum = value
                    min_index = index
                    maximum = value
                    max_index = index
                    break

        if not min_positive:
            for index in range(index, length):
                value = data[index]
                if value > maximum:
                    maximum = value
                    max_index = index
                elif value < minimum:
                    minimum = value
                    min_index = index

        else:
            # Loop until min_pos is defined
            for index in range(index, length):
                value = data[index]
                if value > maximum:
                    maximum = value
                    max_index = index
                elif value < minimum:
                    minimum = value
                    min_index = index

                if value > 0:
                    min_pos = value
                    min_pos_index = index
                    break

            # Loop until the end
            for index in range(index+1, length):
                value = data[index]
                if value > maximum:
                    maximum = value
                    max_index = index
                else:
                    if value < minimum:
                        minimum = value
                        min_index = index

                    if value > 0 and value < min_pos:
                        min_pos = value
                        min_pos_index = index

    return _MinMaxResult(minimum,
                         min_pos if min_pos > 0 else None,
                         maximum,
                         min_index,
                         min_pos_index if min_pos > 0 else None,
                         max_index)


@cython.embedsignature(True)
def min_max(data not None, bint min_positive=False):
    """Returns min, max and optionally strictly positive min of data.

    It also computes the indices of first occurence of min/max.

    NaNs are ignored while computing min/max unless all data is NaNs,
    in which case returned min/max are NaNs.

    Examples:

    >>> import numpy
    >>> data = numpy.arange(10)

    Usage as a function returning min and max:

    >>> min_, max_ = min_max(data)

    Usage as a function returning a result object to access all information:

    >>> result = min_max(data)  # Do not get positive min
    >>> result.minimum, result.argmin
    (0, 0)
    >>> result.maximum, result.argmax
    (9, 10)
    >>> result.min_positive, result.argmin_positive  # Not computed
    (None, None)

    Getting strictly positive min information:

    >>> result = min_max(data, min_positive=True)
    >>> result.min_positive, result.argmin_positive  # Computed
    (1, 1)

    :param data: Array-like dataset
    :param bool min_positive: True to compute the positive min and argmin
                              Default: False.
    :returns: An object with minimum, maximum and min_positive attributes
              and the indices of first occurence in the flattened data:
              argmin, argmax and argmin_positive attributes.
              If all data is <= 0 or min_positive argument is False, then
              min_positive and argmin_positive are None.
    :raises: ValueError if data is empty
    """
    return _min_max(numpy.asanyarray(data).ravel(), min_positive)


### Mean + Std combo ###

class _MeanStdResult(object):
    """Object storing result from :func:`mean_std`"""

    def __init__(self, mean, std, var, length, ddof):
        self._mean = mean
        self._std = std
        self._var = var
        self._length = length
        self._ddof = ddof

    mean = property(lambda self: self._mean, doc="Mean of the array")

    std = property(lambda self: self._std,
                   doc="Estimation of the standard deviation of the array")

    var = property(lambda self: self._var,
                   doc="Estimation of the variance of the array")

    length = property(lambda self: self._length,
                      doc="Number of elements that where processed")

    ddof = property(lambda self: self._ddof,
                    doc="Means Delta Degrees of Freedom provided to mean_std")

    def __getitem__(self, key):
        if key == 0:
            return self.mean
        elif key == 1:
            return self.std
        else:
            raise IndexError("Index out of range")


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _mean_std_f32(_number[:] data, unsigned int ddof):
    """See :func:`mean_std` for documentation."""
    cdef:
        unsigned int length, index
        float value, mean, M2, delta
        float variance, standard_deviation

    length = len(data)

    if length == 0:
        raise ValueError('Zero-size array')

    mean = 0
    M2 = 0

    for index in range(length):
        value = data[index]
        delta = value - mean
        mean += delta / (index + 1)
        M2 += delta * (value - mean)

    if length <= ddof:
        variance = NAN
        standard_deviation = NAN
    else:
        variance = M2 / (length - ddof)
        standard_deviation = sqrt(variance)

    return _MeanStdResult(mean, standard_deviation, variance, length, ddof)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _mean_std_f64(_number[:] data, unsigned int ddof):
    """See :func:`mean_std` for documentation."""
    cdef:
        unsigned int length, index
        double value, mean, M2, delta
        double variance, standard_deviation

    length = len(data)

    if length == 0:
        raise ValueError('Zero-size array')

    mean = 0
    M2 = 0

    for index in range(length):
        value = data[index]
        delta = value - mean
        mean += delta / (index + 1)
        M2 += delta * (value - mean)

    if length <= ddof:
        variance = NAN
        standard_deviation = NAN
    else:
        variance = M2 / (length - ddof)
        standard_deviation = sqrt(variance)

    return _MeanStdResult(mean, standard_deviation, variance, length, ddof)


@cython.embedsignature(True)
def mean_std(data not None, dtype=None, unsigned int ddof=0):
    """Computes mean and estimation of std and variance in a single pass.

    NaNs are propagated.
    Behavior with inf values differs from numpy equivalent functions.

    Examples:

    >>> import numpy
    >>> data = numpy.arange(100.)

    Usage as a function returning mean and std:

    >>> mean, std = mean_std(data)

    Usage as a function returning a result object to access all information:

    >>> result = mean_std(data)
    >>> result.mean, result.var
    (49.5, 833.25)
    >>> result.length, result.ddof
    (100, 0)

    See:

    - http://www.johndcook.com/blog/standard_deviation/
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Welford, B. P.
    Not on a method for calculating corrected sums of squares and products
    Technometrics, Vol. 4, No. 3 (August 1962), pp. 419-420
    American Statisical Association and American Society for Quality.
    DOI: 10.2307/1266577

    :param data: Array-like dataset
    :param dtype:
        Type to use in computing mean, std and variance.
        Default: ``'float64'`` for integers, data type for floating data.
        Only ``'float32'`` and ``'float64'`` types are valid.
    :param int ddof:
        Means Delta Degrees of Freedom.
        The divisor used in calculations is ``(data.size - ddof)``.
        Default: 0 (as in ``numpy.std``).
    :returns: An object with mean, std and var attributes
    :raises: ValueError if data is empty
    """
    cdef:
        bint is_double = True

    data = numpy.asanyarray(data).ravel()

    # Select the dtype to use to compute mean and std
    if dtype is None:
        if data.dtype.name == 'float32':
            is_double = False
    else:
        dtype = numpy.dtype(dtype)
        assert dtype.kind == 'f'
        if dtype.name != 'float64':
            is_double = False

    if is_double:
        return _mean_std_f64(data, ddof)
    else:
        return _mean_std_f32(data, ddof)
