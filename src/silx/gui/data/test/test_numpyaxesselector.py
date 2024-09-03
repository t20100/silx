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
__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/01/2018"

import numpy

from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector
from silx.gui.utils.testutils import SignalListener
from silx.gui.utils.testutils import TestCaseQt

import h5py


class TestNumpyAxesSelector(TestCaseQt):
    def test_creation(self):
        data = numpy.arange(3 * 3 * 3)
        data.shape = 3, 3, 3
        widget = NumpyAxesSelector()
        widget.setVisible(True)

    def test_none(self):
        data = numpy.arange(3 * 3 * 3)
        widget = NumpyAxesSelector()
        widget.setData(data)
        widget.setData(None)
        result = widget.selectedData()
        self.assertIsNone(result)

    def test_output_samedim(self):
        data = numpy.arange(3 * 3 * 3)
        data.shape = 3, 3, 3
        expectedResult = data

        widget = NumpyAxesSelector()
        widget.setAxisNames(["x", "y", "z"])
        widget.setData(data)
        result = widget.selectedData()
        self.assertTrue(numpy.array_equal(result, expectedResult))

    def test_output_moredim(self):
        data = numpy.arange(3 * 3 * 3 * 3)
        data.shape = 3, 3, 3, 3
        expectedResult = data

        widget = NumpyAxesSelector()
        widget.setAxisNames(["x", "y", "z", "boum"])
        widget.setData(data[0])
        result = widget.selectedData()
        self.assertIsNone(result)
        widget.setData(data)
        result = widget.selectedData()
        self.assertTrue(numpy.array_equal(result, expectedResult))

    def test_output_lessdim(self):
        data = numpy.arange(3 * 3 * 3)
        data.shape = 3, 3, 3
        expectedResult = data[0]

        widget = NumpyAxesSelector()
        widget.setAxisNames(["y", "x"])
        widget.setData(data)
        result = widget.selectedData()
        self.assertTrue(numpy.array_equal(result, expectedResult))

    def test_output_1dim(self):
        data = numpy.arange(3 * 3 * 3)
        data.shape = 3, 3, 3
        expectedResult = data[0, 0, 0]

        widget = NumpyAxesSelector()
        widget.setData(data)
        result = widget.selectedData()
        self.assertTrue(numpy.array_equal(result, expectedResult))

    def test_data_event(self):
        data = numpy.arange(3 * 3 * 3)
        widget = NumpyAxesSelector()
        listener = SignalListener()
        widget.dataChanged.connect(listener)
        widget.setData(data)
        widget.setData(None)
        self.assertEqual(listener.callCount(), 2)

    def test_selected_data_event(self):
        data = numpy.arange(3 * 3 * 3)
        data.shape = 3, 3, 3
        widget = NumpyAxesSelector()
        listener = SignalListener()
        widget.selectionChanged.connect(listener)
        widget.setData(data)
        widget.setAxisNames(["x"])
        widget.setData(None)
        self.assertEqual(listener.callCount(), 3)
        listener.clear()


def test_h5py_dataset(tmp_path, qWidgetFactory):
    widget = qWidgetFactory(NumpyAxesSelector)

    with h5py.File(tmp_path / "test.h5", "w") as h5file:
        h5file["data"] = numpy.arange(3 * 3 * 3).reshape(3, 3, 3)

        widget.setData(h5file["data"])
        widget.setAxisNames(["y", "x"])

        result = widget.selectedData()
        expectedResult = h5file["data"][0]
        assert numpy.array_equal(result, expectedResult)