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
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "05/12/2016"

import numpy

from silx.gui import qt
from silx.gui.data import ArrayTableWidget
from silx.gui.data.ArrayTableModel import ArrayTableModel
from silx.gui.utils.testutils import TestCaseQt

import h5py
import pytest


class TestArrayWidget(TestCaseQt):
    """Basic test for ArrayTableWidget with a numpy array"""

    def setUp(self):
        super(TestArrayWidget, self).setUp()
        self.aw = ArrayTableWidget.ArrayTableWidget()

    def tearDown(self):
        del self.aw
        super(TestArrayWidget, self).tearDown()

    def testShow(self):
        """test for errors"""
        self.aw.show()
        self.qWaitForWindowExposed(self.aw)

    def testSetData0D(self):
        a = 1
        self.aw.setArrayData(a)
        b = self.aw.getData(copy=True)

        self.assertTrue(numpy.array_equal(a, b))

        # scalar/0D data has no frame index
        self.assertEqual(len(self.aw.model._index), 0)
        # and no perspective
        self.assertEqual(len(self.aw.model._perspective), 0)

    def testSetData1D(self):
        a = [1, 2]
        self.aw.setArrayData(a)
        b = self.aw.getData(copy=True)

        self.assertTrue(numpy.array_equal(a, b))

        # 1D data has no frame index
        self.assertEqual(len(self.aw.model._index), 0)
        # and no perspective
        self.assertEqual(len(self.aw.model._perspective), 0)

    def testSetData4D(self):
        a = numpy.reshape(numpy.linspace(0.213, 1.234, 1250), (5, 5, 5, 10))
        self.aw.setArrayData(a)

        # default perspective (0, 1)
        self.assertEqual(list(self.aw.model._perspective), [0, 1])
        self.aw.setPerspective((1, 3))
        self.assertEqual(list(self.aw.model._perspective), [1, 3])

        b = self.aw.getData(copy=True)
        self.assertTrue(numpy.array_equal(a, b))

        # 4D data has a 2-tuple as frame index
        self.assertEqual(len(self.aw.model._index), 2)
        # default index is (0, 0)
        self.assertEqual(list(self.aw.model._index), [0, 0])
        self.aw.setFrameIndex((3, 1))

        self.assertEqual(list(self.aw.model._index), [3, 1])

    def testColors(self):
        a = numpy.arange(256, dtype=numpy.uint8)
        self.aw.setArrayData(a)

        bgcolor = numpy.empty(a.shape + (3,), dtype=numpy.uint8)
        # Black & white palette
        bgcolor[..., 0] = a
        bgcolor[..., 1] = a
        bgcolor[..., 2] = a

        fgcolor = numpy.bitwise_xor(bgcolor, 255)

        self.aw.setArrayColors(bgcolor, fgcolor)

        # test colors are as expected in model
        for i in range(256):
            # all RGB channels for BG equal to data value
            self.assertEqual(
                self.aw.model.data(
                    self.aw.model.index(0, i), role=qt.Qt.BackgroundRole
                ),
                qt.QColor(i, i, i),
                "Unexpected background color",
            )

            # all RGB channels for FG equal to XOR(data value, 255)
            self.assertEqual(
                self.aw.model.data(
                    self.aw.model.index(0, i), role=qt.Qt.ForegroundRole
                ),
                qt.QColor(i ^ 255, i ^ 255, i ^ 255),
                "Unexpected text color",
            )

        # test colors are reset to None when a new data array is loaded
        # with different shape
        self.aw.setArrayData(numpy.arange(300))

        for i in range(300):
            # all RGB channels for BG equal to data value
            self.assertIsNone(
                self.aw.model.data(self.aw.model.index(0, i), role=qt.Qt.BackgroundRole)
            )

    def testDefaultFlagNotEditable(self):
        """editable should be False by default, in setArrayData"""
        self.aw.setArrayData([[0]])
        idx = self.aw.model.createIndex(0, 0)
        # model is editable
        self.assertFalse(self.aw.model.flags(idx) & qt.Qt.ItemIsEditable)

    def testFlagEditable(self):
        self.aw.setArrayData([[0]], editable=True)
        idx = self.aw.model.createIndex(0, 0)
        # model is editable
        self.assertTrue(self.aw.model.flags(idx) & qt.Qt.ItemIsEditable)

    def testFlagNotEditable(self):
        self.aw.setArrayData([[0]], editable=False)
        idx = self.aw.model.createIndex(0, 0)
        # model is editable
        self.assertFalse(self.aw.model.flags(idx) & qt.Qt.ItemIsEditable)

    def testReferenceReturned(self):
        """when setting the data with copy=False and
        retrieving it with getData(copy=False), we should recover
        the same original object.
        """
        # n-D (n >=2)
        a0 = numpy.reshape(numpy.linspace(0.213, 1.234, 1000), (10, 10, 10))
        self.aw.setArrayData(a0, copy=False)
        a1 = self.aw.getData(copy=False)

        self.assertIs(a0, a1)

        # 1D
        b0 = numpy.linspace(0.213, 1.234, 1000)
        self.aw.setArrayData(b0, copy=False)
        b1 = self.aw.getData(copy=False)
        self.assertIs(b0, b1)

    def testClipping(self):
        """Test clipping of large arrays"""
        self.aw.show()
        self.qWaitForWindowExposed(self.aw)

        data = numpy.arange(ArrayTableModel.MAX_NUMBER_OF_SECTIONS + 10)

        for shape in [(1, -1), (-1, 1)]:
            with self.subTest(shape=shape):
                self.aw.setArrayData(data.reshape(shape), editable=True)
                self.qapp.processEvents()


def testReadOnly(tmp_path, qWidgetFactory):
    """Open H5 dataset in read-only mode, ensure the model is not editable."""
    data = numpy.reshape(numpy.linspace(0.213, 1.234, 1000), (10, 10, 10))
    with h5py.File(tmp_path / "array.h5", mode="w") as h5f:
        h5f["my_array"] = data

    arrayTableWidget = qWidgetFactory(ArrayTableWidget.ArrayTableWidget)

    with h5py.File(tmp_path / "array.h5", "r") as h5f:
        # ArrayTableModel relies on following condition
        assert h5f["my_array"].file.mode == "r"

        arrayTableWidget.setArrayData(h5f["my_array"], copy=False, editable=True)

        assert isinstance(h5f["my_array"], h5py.Dataset)  # simple sanity check
        # internal representation must be a reference to original data (copy=False)
        assert isinstance(arrayTableWidget.model._array, h5py.Dataset)
        assert arrayTableWidget.model._array.file.mode == "r"

        assert numpy.array_equal(data, arrayTableWidget.getData())

        # model must have detected read-only dataset and disabled editing
        assert not arrayTableWidget.model._editable
        idx = arrayTableWidget.model.createIndex(0, 0)
        assert not (arrayTableWidget.model.flags(idx) & qt.Qt.ItemIsEditable)

        # force editing read-only datasets raises IOError
        with pytest.raises(IOError):
            arrayTableWidget.model.setData(idx, 123.4, role=qt.Qt.EditRole)


def testReadWrite(tmp_path, qWidgetFactory):
    data = numpy.reshape(numpy.linspace(0.213, 1.234, 1000), (10, 10, 10))
    with h5py.File(tmp_path / "array.h5", mode="w") as h5f:
        h5f["my_array"] = data

    arrayTableWidget = qWidgetFactory(ArrayTableWidget.ArrayTableWidget)

    with h5py.File(tmp_path / "array.h5", "r+") as h5f:
        assert h5f["my_array"].file.mode == "r+"

        arrayTableWidget.setArrayData(h5f["my_array"], copy=False, editable=True)
        assert numpy.array_equal(data, arrayTableWidget.getData(copy=False))

        idx = arrayTableWidget.model.createIndex(0, 0)
        # model is editable
        assert (arrayTableWidget.model.flags(idx) & qt.Qt.ItemIsEditable)


def testSetData0D(tmp_path, qWidgetFactory):
    with h5py.File(tmp_path / "array.h5", mode="w") as h5f:
        h5f["my_scalar"] = 3.14

    arrayTableWidget = qWidgetFactory(ArrayTableWidget.ArrayTableWidget)

    with h5py.File(tmp_path / "array.h5", "r+") as h5f:
        arrayTableWidget.setArrayData(h5f["my_scalar"])
        assert numpy.array_equal(h5f["my_scalar"], arrayTableWidget.getData(copy=True))


def testSetData1D(tmp_path, qWidgetFactory):
    with h5py.File(tmp_path / "array.h5", mode="w") as h5f:
        h5f["my_1D_array"] = numpy.array(numpy.arange(1000))

    arrayTableWidget = qWidgetFactory(ArrayTableWidget.ArrayTableWidget)

    with h5py.File(tmp_path / "array.h5", "r+") as h5f:
        arrayTableWidget.setArrayData(h5f["my_1D_array"])

        assert numpy.array_equal(h5f["my_1D_array"], arrayTableWidget.getData(copy=True))


def testReferenceReturned(tmp_path, qWidgetFactory):
    """when setting the data with copy=False and
    retrieving it with getData(copy=False), we should recover
    the same original object.

    This only works for array with at least 2D. For 1D and 0D
    arrays, a view is created at some point, which  in the case
    of an hdf5 dataset creates a copy."""
    data = numpy.reshape(numpy.linspace(0.213, 1.234, 1000), (10, 10, 10))
    with h5py.File(tmp_path / "array.h5", mode="w") as h5f:
        h5f["my_array"] = data
        h5f["my_1D_array"] = numpy.array(numpy.arange(1000))

    arrayTableWidget = qWidgetFactory(ArrayTableWidget.ArrayTableWidget)

    with h5py.File(tmp_path / "array.h5", "r+") as h5f:
        # n-D
        arrayTableWidget.setArrayData(h5f["my_array"], copy=False)
        assert h5f["my_array"] is arrayTableWidget.getData(copy=False)

        # 1D
        arrayTableWidget.setArrayData(h5f["my_1D_array"], copy=False)
        assert h5f["my_1D_array"] is arrayTableWidget.getData(copy=False)
