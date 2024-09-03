# /*##########################################################################
#
# Copyright (c) 2016-2024 European Synchrotron Radiation Facility
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
"""Tests for array_like module"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "09/01/2017"


from itertools import permutations
import h5py
import numpy
import unittest

from ..array_like import DatasetView, ListOfImages
from ..array_like import (
    get_dtype,
    get_concatenated_dtype,
    get_shape,
    is_array,
    is_nested_sequence,
    is_list_of_arrays,
)


import pytest


VOLUME = numpy.arange(5 * 10 * 20).reshape(5, 10, 20)


@pytest.fixture(scope="module")
def h5VolumeDataset(tmp_path):
    """Fixture providing a h5py.Dataset containing the VOLUME data"""
    h5path = tmp_path / "tempfile.h5"
    with h5py.File(h5path, "w") as h5f:
        h5f["volume"] = VOLUME
    with h5py.File(h5path, "r") as h5f:
        yield h5f["volume"]


def checkVolumeSize(obj):
    """These assertions apply to all following test cases"""
    assert obj.ndim == VOLUME.ndim
    assert obj.size == VOLUME.size
    size_from_shape = 1
    for dim in obj.shape:
        size_from_shape *= dim
    assert size_from_shape == VOLUME.size

    for dim in VOLUME.shape:
        assert dim in obj.shape



def testNoTransposition(h5VolumeDataset):
    """no transposition (transposition = (0, 1, 2))"""
    a = DatasetView(h5VolumeDataset)

    assert a.shape == VOLUME.shape
    checkVolumeSize(a)

    # reversing the dimensions twice results in no change
    rtrans = list(reversed(range(VOLUME.ndim)))
    assert numpy.array_equal(a, a.transpose(rtrans).transpose(rtrans))

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                assert h5VolumeDataset[i, j, k] == a[i, j, k]


@pytest.mark.parametrize("transposition", list(permutations([0, 1, 2])))
def testTransposition(h5VolumeDataset, transposition):
    """test transposed dataset

    :param tuple transposition: List of dimensions (0... n-1) sorted
        in the desired order
    """
    a = DatasetView(h5VolumeDataset, transposition=transposition)
    checkVolumeSize(a)

    # sort shape of transposed object, to hopefully find the original shape
    sorted_shape = tuple(
        dim_size for (_, dim_size) in sorted(zip(transposition, a.shape))
    )
    assert sorted_shape == VOLUME.shape

    a_as_array = numpy.array(h5VolumeDataset).transpose(transposition)

    # test the __array__ method
    assert numpy.array_equal(numpy.array(a), a_as_array)

    # test slicing
    for selection in [
        (2, slice(None), slice(None)),
        (slice(None), 1, slice(0, 8)),
        (slice(0, 3), slice(None), 3),
        (1, 3, slice(None)),
        (slice(None), 2, 1),
        (4, slice(1, 9, 2), 2),
    ]:
        assert isinstance(a[selection], numpy.ndarray)
        assert numpy.array_equal(a[selection], a_as_array[selection])

    # test the DatasetView.__getitem__ for single values
    # (step adjusted to test at least 3 indices in each dimension)
    for i in range(0, a.shape[0], a.shape[0] // 3):
        for j in range(0, a.shape[1], a.shape[1] // 3):
            for k in range(0, a.shape[2], a.shape[2] // 3):
                sorted_indices = tuple(
                    idx for (_, idx) in sorted(zip(transposition, [i, j, k]))
                )
                viewed_value = a[i, j, k]
                corresponding_original_value = h5VolumeDataset[sorted_indices]
                assert viewed_value == corresponding_original_value

    # reversing the dimensions twice results in no change
    rtrans = list(reversed(range(VOLUME.ndim)))
    assert numpy.array_equal(a, a.transpose(rtrans).transpose(rtrans))

    # test .T property
    assert numpy.array_equal(a.T, a.transpose(rtrans))


@pytest.mark.parametrize("transposition1", list(permutations([0, 1, 2])))
@pytest.mark.parametrize("transposition2", list(permutations([0, 1, 2])))
def testDoubleTransposition(h5VolumeDataset, transposition1, transposition2):
    a = DatasetView(h5VolumeDataset, transposition=transposition1).transpose(
        transposition2
    )

    b = VOLUME.transpose(transposition1).transpose(transposition2)

    assert numpy.array_equal(a, b), f"failed with double transposition {transposition1} {transposition2}"


def test1DIndex(h5VolumeDataset):
    a = DatasetView(h5VolumeDataset)
    assert numpy.array_equal(VOLUME[1], a[1])

    b = DatasetView(h5VolumeDataset, transposition=(1, 0, 2))
    assert numpy.array_equal(VOLUME[:, 1, :], b[1])


class TestTransposedListOfImages(unittest.TestCase):
    def setUp(self):
        # images attributes
        self.ndim = 3
        self.original_shape = (5, 10, 20)
        self.size = 1
        for dim in self.original_shape:
            self.size *= dim

        volume = numpy.arange(self.size).reshape(self.original_shape)

        self.images = []
        for i in range(self.original_shape[0]):
            self.images.append(volume[i])

        self.images_as_3D_array = numpy.array(self.images)

        self.all_permutations = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]

    def tearDown(self):
        pass

    def _testSize(self, obj):
        """These assertions apply to all following test cases"""
        self.assertEqual(obj.ndim, self.ndim)
        self.assertEqual(obj.size, self.size)
        size_from_shape = 1
        for dim in obj.shape:
            size_from_shape *= dim
        self.assertEqual(size_from_shape, self.size)

        for dim in self.original_shape:
            self.assertIn(dim, obj.shape)

    def testNoTransposition(self):
        """no transposition (transposition = (0, 1, 2))"""
        a = ListOfImages(self.images)

        self.assertEqual(a.shape, self.original_shape)
        self._testSize(a)

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    self.assertEqual(self.images[i][j, k], a[i, j, k])

        # reversing the dimensions twice results in no change
        rtrans = list(reversed(range(self.ndim)))
        self.assertTrue(numpy.array_equal(a, a.transpose(rtrans).transpose(rtrans)))

        # test .T property
        self.assertTrue(numpy.array_equal(a.T, a.transpose(rtrans)))

    def _testTransposition(self, transposition):
        """test transposed dataset

        :param tuple transposition: List of dimensions (0... n-1) sorted
            in the desired order
        """
        a = ListOfImages(self.images, transposition=transposition)
        self._testSize(a)

        # sort shape of transposed object, to hopefully find the original shape
        sorted_shape = tuple(
            dim_size for (_, dim_size) in sorted(zip(transposition, a.shape))
        )
        self.assertEqual(sorted_shape, self.original_shape)

        a_as_array = numpy.array(self.images).transpose(transposition)

        # test the DatasetView.__array__ method
        self.assertTrue(numpy.array_equal(numpy.array(a), a_as_array))

        # test slicing
        for selection in [
            (2, slice(None), slice(None)),
            (slice(None), 1, slice(0, 8)),
            (slice(0, 3), slice(None), 3),
            (1, 3, slice(None)),
            (slice(None), 2, 1),
            (4, slice(1, 9, 2), 2),
        ]:
            self.assertIsInstance(a[selection], numpy.ndarray)
            self.assertTrue(numpy.array_equal(a[selection], a_as_array[selection]))

        # test the DatasetView.__getitem__ for single values
        # (step adjusted to test at least 3 indices in each dimension)
        for i in range(0, a.shape[0], a.shape[0] // 3):
            for j in range(0, a.shape[1], a.shape[1] // 3):
                for k in range(0, a.shape[2], a.shape[2] // 3):
                    viewed_value = a[i, j, k]
                    sorted_indices = tuple(
                        idx for (_, idx) in sorted(zip(transposition, [i, j, k]))
                    )
                    corresponding_original_value = self.images[sorted_indices[0]][
                        sorted_indices[1:]
                    ]
                    self.assertEqual(viewed_value, corresponding_original_value)

        # reversing the dimensions twice results in no change
        rtrans = list(reversed(range(self.ndim)))
        self.assertTrue(numpy.array_equal(a, a.transpose(rtrans).transpose(rtrans)))

        # test .T property
        self.assertTrue(numpy.array_equal(a.T, a.transpose(rtrans)))

    def _testDoubleTransposition(self, transposition1, transposition2):
        a = ListOfImages(self.images, transposition=transposition1).transpose(
            transposition2
        )

        b = self.images_as_3D_array.transpose(transposition1).transpose(transposition2)

        self.assertTrue(
            numpy.array_equal(a, b),
            "failed with double transposition %s %s" % (transposition1, transposition2),
        )

    def testTransposition012(self):
        """transposition = (0, 1, 2)
        (should be the same as testNoTransposition)"""
        self._testTransposition((0, 1, 2))

    def testTransposition021(self):
        """transposition = (0, 2, 1)"""
        self._testTransposition((0, 2, 1))

    def testTransposition102(self):
        """transposition = (1, 0, 2)"""
        self._testTransposition((1, 0, 2))

    def testTransposition120(self):
        """transposition = (1, 2, 0)"""
        self._testTransposition((1, 2, 0))

    def testTransposition201(self):
        """transposition = (2, 0, 1)"""
        self._testTransposition((2, 0, 1))

    def testTransposition210(self):
        """transposition = (2, 1, 0)"""
        self._testTransposition((2, 1, 0))

    def testAllDoubleTranspositions(self):
        for trans1 in self.all_permutations:
            for trans2 in self.all_permutations:
                self._testDoubleTransposition(trans1, trans2)

    def test1DIndex(self):
        a = ListOfImages(self.images)
        self.assertTrue(numpy.array_equal(self.images[1], a[1]))

        b = ListOfImages(self.images, transposition=(1, 0, 2))
        self.assertTrue(numpy.array_equal(self.images_as_3D_array[:, 1, :], b[1]))


def testFunctionsWithListOfLists():
    """Test functions to guess the dtype and shape of an array_like
    object"""
    l = [[0, 1, 2], [2, 3, 4]]
    assert get_dtype(l) == numpy.dtype(int)
    assert get_shape(l) == (2, 3)
    assert is_nested_sequence(l)
    assert not is_array(l)
    assert not is_list_of_arrays(l)

    l = [[0.0, 1.0], [2.0, 3.0]]
    assert get_dtype(l) == numpy.dtype(float)
    assert get_shape(l) ==  (2, 2)
    assert is_nested_sequence(l)
    assert not is_array(l)
    assert not is_list_of_arrays(l)

    # concatenated dtype of int and float
    l = [
        numpy.array([[0, 1, 2], [2, 3, 4]]),
        numpy.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
    ]

    assert get_concatenated_dtype(l) == numpy.array(l).dtype
    assert get_shape(l) == (2, 2, 3)
    assert not is_nested_sequence(l)
    assert not is_array(l)
    assert is_list_of_arrays(l)


def testFunctionsWithNumpyArray():
    array = numpy.array([[0, 1], [2, 3]])
    assert get_dtype(array) == array.dtype
    assert not is_nested_sequence(array)
    assert is_array(array)
    assert not is_list_of_arrays(array)


def testFunctionsWithH5pyDataset(tmp_path):
    with h5py.File(tmp_path / "tempfile.h5", "w") as h5f:
        h5f["dataset"] = numpy.array([[0, 1], [2, 3]])

        dataset = h5f["dataset"]

        assert get_dtype(dataset) == numpy.dtype(int)
        assert not is_nested_sequence(dataset)
        assert is_array(dataset)
        assert not is_list_of_arrays(dataset)
