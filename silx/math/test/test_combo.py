# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Tests of the combo module"""

from __future__ import division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "20/12/2016"


import unittest

import numpy

from silx.test.utils import ParametricTestCase

from silx.math.combo import min_max, mean_std


FLOATING_DTYPES = 'float32', 'float64'
SIGNED_INT_DTYPES = 'uint8', 'uint16', 'uint32', 'uint64'
UNSIGNED_INT_DTYPES = 'uint8', 'uint16', 'uint32', 'uint64'
DTYPES = FLOATING_DTYPES + SIGNED_INT_DTYPES + UNSIGNED_INT_DTYPES


class TestMinMax(ParametricTestCase):
    """Tests of min max combo"""

    def _test_min_max(self, data, min_positive):
        """Compare min_max with numpy for the given dataset

        :param numpy.ndarray data: Data set to use for test
        :param bool min_positive: True to test with positive min
        """
        result = min_max(data, min_positive)

        minimum = numpy.nanmin(data)
        if numpy.isnan(minimum):  # All NaNs
            self.assertTrue(numpy.isnan(result.minimum))
            self.assertEqual(result.argmin, 0)

        else:
            self.assertEqual(result.minimum, minimum)

            argmin = numpy.where(data == minimum)[0][0]
            self.assertEqual(result.argmin, argmin)

        maximum = numpy.nanmax(data)
        if numpy.isnan(maximum):  # All NaNs
            self.assertTrue(numpy.isnan(result.maximum))
            self.assertEqual(result.argmax, 0)

        else:
            self.assertEqual(result.maximum, maximum)

            argmax = numpy.where(data == maximum)[0][0]
            self.assertEqual(result.argmax, argmax)

        if min_positive:
            pos_data = data[data > 0]
            if len(pos_data) > 0:
                min_pos = numpy.min(pos_data)
                argmin_pos = numpy.where(data == min_pos)[0][0]
            else:
                min_pos = None
                argmin_pos = None
            self.assertEqual(result.min_positive, min_pos)
            self.assertEqual(result.argmin_positive, argmin_pos)

    def test_different_datasets(self):
        """Test min_max with different numpy.arange datasets."""
        size = 1000

        for dtype in DTYPES:

            tests = {
                '0 to N': (0, 1),
                'N-1 to 0': (size - 1, -1)}
            if dtype not in UNSIGNED_INT_DTYPES:
                tests['N/2 to -N/2'] = size // 2, -1
                tests['0 to -N'] = 0, -1

            for name, (start, step) in tests.items():
                for min_positive in (True, False):
                    with self.subTest(dtype=dtype,
                                      min_positive=min_positive,
                                      data=name):
                        data = numpy.arange(
                            start, start + step * size, step, dtype=dtype)

                        self._test_min_max(data, min_positive)

    def test_nodata(self):
        """Test min_max with None and empty array"""
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    min_max(None)
                
                data = numpy.array((), dtype=dtype)
                with self.assertRaises(ValueError):
                    min_max(data)

    def test_nandata(self):
        """Test min_max with NaN in data"""
        tests = [
            (float('nan'), float('nan')),  # All NaNs
            (float('nan'), 1.0),  # NaN first and positive
            (float('nan'), -1.0),  # NaN first and negative
            (1.0, 2.0, float('nan')),  # NaN last and positive
            (-1.0, -2.0, float('nan')),  # NaN last and negative
            (1.0, float('nan'), -1.0),  # Some NaN
        ]

        for dtype in FLOATING_DTYPES:
            for data in tests:
                with self.subTest(dtype=dtype, data=data):
                    data = numpy.array(data, dtype=dtype)
                    self._test_min_max(data, min_positive=True)

    def test_infdata(self):
        """Test min_max with inf."""
        tests = [
            [float('inf')] * 3,  # All +inf
            [float('inf')] * 3,  # All -inf
            (float('inf'), float('-inf')),  # + and - inf
            (float('inf'), float('-inf'), float('nan')),  # +/-inf, nan last
            (float('nan'), float('-inf'), float('inf')),  # +/-inf, nan first
            (float('inf'), float('nan'), float('-inf')),  # +/-inf, nan center
        ]

        for dtype in FLOATING_DTYPES:
            for data in tests:
                with self.subTest(dtype=dtype, data=data):
                    data = numpy.array(data, dtype=dtype)
                    self._test_min_max(data, min_positive=True)


class TestMeanStd(ParametricTestCase):
    """Test mean_std combo against numpy"""

    def _test_mean_std(self, data, ddof):
        """Compare mean_std with numpy for the given dataset

        :param numpy.ndarray data: Data set to use for test
        :param int ddof: Means Delta Degrees of Freedom std argument
        """
        result = mean_std(data, ddof=ddof)
        self.assertEqual(result.ddof, ddof)
        self.assertEqual(result.length, data.size)

        mean = numpy.mean(data)
        if numpy.isnan(mean):
            self.assertTrue(numpy.isnan(result.mean))
        else:
            self.assertEqual(result.mean, mean)

        if numpy.any(numpy.isnan(data)) or len(data) <= ddof:
            self.assertTrue(numpy.isnan(result.std))
            self.assertTrue(numpy.isnan(result.var))

        else:
            std = numpy.std(data, ddof=ddof)
            self.assertTrue(numpy.allclose(result.std, std))

            var = numpy.var(data, ddof=ddof)
            self.assertTrue(numpy.allclose(result.var, var))

    def test(self):
        """Test Mean_std with different datasets"""

        for dtype in FLOATING_DTYPES:
            for ddof in (0., 1.):
                with self.subTest(dtype=dtype, ddof=ddof):
                    data = numpy.arange(1000., dtype=dtype)
                    self._test_mean_std(data, ddof)

    def test_no_data(self):
        """Test mean_std without data of with too small data"""
        for dtype in FLOATING_DTYPES:
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    mean_std(None)

                data = numpy.array((), dtype=dtype)
                with self.assertRaises(ValueError):
                    mean_std(data)

                # Not enough data for ddof
                data = numpy.array((1.,), dtype=dtype)
                self._test_mean_std(data, ddof=1)

    def test_nan_data(self):
        """Test min_max with NaN in data"""
        tests = [
            (float('nan'), float('nan')),  # All NaNs
            (2.0, float('nan'), 1.0),  # Some NaN
        ]

        for dtype in FLOATING_DTYPES:
            for data in tests:
                with self.subTest(dtype=dtype, data=data):
                    data = numpy.array(data, dtype=dtype)
                    self._test_mean_std(data, ddof=0)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestMinMax))
    test_suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestMeanStd))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
