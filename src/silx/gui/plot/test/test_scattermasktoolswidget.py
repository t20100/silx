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
"""Basic tests for MaskToolsWidget"""

__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "17/01/2018"


import numpy

from silx.gui import qt
from silx.gui.utils.testutils import getQToolButtonFromAction
from silx.gui.plot import PlotWindow, ScatterMaskToolsWidget

import pytest
from .test_masktoolswidget import _drag, _drawPolygon, _drawPencil  # TODO


def testEmptyPlot(qapp, qWidgetFactory):
    """Empty plot, display MaskToolsDockWidget, toggle multiple masks"""
    plot = qWidgetFactory(PlotWindow)
    maskDock = ScatterMaskToolsWidget.ScatterMaskToolsDockWidget(plot=plot, name="TEST")
    plot.addDockWidget(qt.Qt.BottomDockWidgetArea, maskDock)
    maskWidget = maskDock.widget()

    maskWidget.setMultipleMasks("single")
    qapp.processEvents()

    maskWidget.setMultipleMasks("exclusive")
    qapp.processEvents()


def testWithAScatter(qapp, qapp_utils, qWidgetFactory):
    """Plot with a Scatter: test MaskToolsWidget interactions"""
    plot = qWidgetFactory(PlotWindow)
    maskDock = ScatterMaskToolsWidget.ScatterMaskToolsDockWidget(plot=plot, name="TEST")
    plot.addDockWidget(qt.Qt.BottomDockWidgetArea, maskDock)
    maskWidget = maskDock.widget()

    # Add and remove a scatter (this should enable/disable GUI + change mask)
    plot.addScatter(
        x=numpy.arange(256),
        y=numpy.arange(256),
        value=numpy.random.random(256),
        legend="test",
    )
    plot.setActiveScatter("test")
    qapp.processEvents()

    plot.remove("test", kind="scatter")
    qapp.processEvents()

    plot.addScatter(
        x=numpy.arange(1000),
        y=1000 * (numpy.arange(1000) % 20),
        value=numpy.random.random(1000),
        legend="test",
    )
    plot.setActiveScatter("test")
    plot.resetZoom()
    qapp.processEvents()

    # Test draw rectangle #
    toolButton = getQToolButtonFromAction(maskWidget.rectAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

    # mask
    maskWidget.maskStateGroup.button(1).click()
    qapp.processEvents()
    _drag(qapp, qapp_utils, plot)

    assert not numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))

    # unmask same region
    maskWidget.maskStateGroup.button(0).click()
    qapp.processEvents()
    _drag(qapp, qapp_utils, plot)
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))

    # Test draw polygon #
    toolButton = getQToolButtonFromAction(maskWidget.polygonAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

    # mask
    maskWidget.maskStateGroup.button(1).click()
    qapp.processEvents()
    _drawPolygon(qapp, qapp_utils, plot)
    assert not numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))

    # unmask same region
    maskWidget.maskStateGroup.button(0).click()
    qapp.processEvents()
    _drawPolygon(qapp, qapp_utils, plot)
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))

    # Test draw pencil #
    toolButton = getQToolButtonFromAction(maskWidget.pencilAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

    maskWidget.pencilSpinBox.setValue(30)
    qapp.processEvents()

    # mask
    maskWidget.maskStateGroup.button(1).click()
    qapp.processEvents()
    _drawPencil(qapp, qapp_utils, plot)
    assert not numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))

    # unmask same region
    maskWidget.maskStateGroup.button(0).click()
    qapp.processEvents()
    _drawPencil(qapp, qapp_utils, plot)
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))

    # Test no draw tool #
    toolButton = getQToolButtonFromAction(maskWidget.browseAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

    plot.clear()


@pytest.mark.parametrize("file_format", ["npy", "csv"])
def testLoadSave(file_format, tmp_path, qapp, qapp_utils, qWidgetFactory):
    plot = qWidgetFactory(PlotWindow)
    maskDock = ScatterMaskToolsWidget.ScatterMaskToolsDockWidget(plot=plot, name="TEST")
    plot.addDockWidget(qt.Qt.BottomDockWidgetArea, maskDock)
    maskWidget = maskDock.widget()

    plot.addScatter(
        x=numpy.arange(256),
        y=25 * (numpy.arange(256) % 10),
        value=numpy.random.random(256),
        legend="test",
    )
    plot.setActiveScatter("test")
    plot.resetZoom()
    qapp.processEvents()

    # Draw a polygon mask
    toolButton = getQToolButtonFromAction(maskWidget.polygonAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)
    _drawPolygon(qapp, qapp_utils, plot)

    ref_mask = maskWidget.getSelectionMask()
    assert not numpy.all(numpy.equal(ref_mask, 0))

    mask_filename = str(tmp_path / f"mask.{file_format}")
    maskWidget.save(mask_filename, file_format)

    maskWidget.resetSelectionMask()
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))

    maskWidget.load(mask_filename)
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), ref_mask))


def testSigMaskChangedEmitted(qapp, qapp_utils, qWidgetFactory):
    plot = qWidgetFactory(PlotWindow)
    maskDock = ScatterMaskToolsWidget.ScatterMaskToolsDockWidget(plot=plot, name="TEST")
    plot.addDockWidget(qt.Qt.BottomDockWidgetArea, maskDock)
    maskWidget = maskDock.widget()

    qapp.processEvents()
    plot.addScatter(
        x=numpy.arange(1000),
        y=1000 * (numpy.arange(1000) % 20),
        value=numpy.ones((1000,)),
        legend="test",
    )
    plot.setActiveScatter("test")
    plot.resetZoom()
    qapp.processEvents()

    plot.remove("test", kind="scatter")
    qapp.processEvents()

    plot.addScatter(
        x=numpy.arange(1000),
        y=1000 * (numpy.arange(1000) % 20),
        value=numpy.random.random(1000),
        legend="test",
    )

    l = []

    def slot():
        l.append(1)

    maskWidget.sigMaskChanged.connect(slot)

    # rectangle mask
    toolButton = getQToolButtonFromAction(maskWidget.rectAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)
    maskWidget.maskStateGroup.button(1).click()
    qapp.processEvents()
    _drag(qapp, qapp_utils, plot)

    assert len(l) > 0
