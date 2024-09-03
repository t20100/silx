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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import numpy

from silx.gui import qt
from silx.gui.utils.testutils import getQToolButtonFromAction
from silx.gui.plot import PlotWindow, MaskToolsWidget

import pytest


def _drag(qapp, qapp_utils, plot):
    """Drag from plot center to offset position"""
    plotBackendWidget = plot.getWidgetHandle()
    xCenter, yCenter = plotBackendWidget.width() // 2, plotBackendWidget.height() // 2
    offset = min(plotBackendWidget.width(), plotBackendWidget.height()) // 10

    pos0 = xCenter, yCenter
    pos1 = xCenter + offset, yCenter + offset

    qapp_utils.mouseMove(plotBackendWidget, pos=(0, 0))
    qapp_utils.mouseMove(plotBackendWidget, pos=pos0)
    qapp.processEvents()
    qapp_utils.mousePress(plotBackendWidget, qt.Qt.LeftButton, pos=pos0)
    qapp.processEvents()
    qapp_utils.mouseMove(plotBackendWidget, pos=(pos0[0] + offset // 2, pos0[1] + offset // 2))
    qapp_utils.mouseMove(plotBackendWidget, pos=pos1)
    qapp.processEvents()
    qapp_utils.mouseRelease(plotBackendWidget, qt.Qt.LeftButton, pos=pos1)
    qapp.processEvents()
    qapp_utils.mouseMove(plotBackendWidget, pos=(0, 0))


def _drawPolygon(qapp, qapp_utils, plot):
    """Draw a star polygon in the plot"""
    plotBackendWidget = plot.getWidgetHandle()
    x, y = plotBackendWidget.width() // 2, plotBackendWidget.height() // 2
    offset = min(plotBackendWidget.width(), plotBackendWidget.height()) // 10

    star = [
        (x, y + offset),
        (x - offset, y - offset),
        (x + offset, y),
        (x - offset, y),
        (x + offset, y - offset),
        (x, y + offset),
    ]  # Close polygon

    qapp_utils.mouseMove(plotBackendWidget, pos=(0, 0))
    for pos in star:
        qapp_utils.mouseMove(plotBackendWidget, pos=pos)
        qapp.processEvents()
        qapp_utils.mousePress(plotBackendWidget, qt.Qt.LeftButton, pos=pos)
        qapp.processEvents()
        qapp_utils.mouseRelease(plotBackendWidget, qt.Qt.LeftButton, pos=pos)
        qapp.processEvents()


def _drawPencil(qapp, qapp_utils, plot):
    """Draw a star polygon in the plot"""
    plotBackendWidget = plot.getWidgetHandle()
    x, y = plotBackendWidget.width() // 2, plotBackendWidget.height() // 2
    offset = min(plotBackendWidget.width(), plotBackendWidget.height()) // 10

    star = [
        (x, y + offset),
        (x - offset, y - offset),
        (x + offset, y),
        (x - offset, y),
        (x + offset, y - offset),
    ]

    qapp_utils.mouseMove(plotBackendWidget, pos=(0, 0))
    for start, end in zip(star[:-1], star[1:]):
        qapp_utils.mouseMove(plotBackendWidget, pos=start)
        qapp_utils.mousePress(plotBackendWidget, qt.Qt.LeftButton, pos=start)
        qapp.processEvents()
        qapp_utils.mouseMove(plotBackendWidget, pos=end)
        qapp.processEvents()
        qapp_utils.mouseRelease(plotBackendWidget, qt.Qt.LeftButton, pos=end)
        qapp.processEvents()


def _isMaskItemSync(maskWidget, plotWidget):
    """Check if masks from item and tools are sync or not"""
    if maskWidget.isItemMaskUpdated():
        return numpy.all(
            numpy.equal(
                maskWidget.getSelectionMask(),
                plotWidget.getActiveImage().getMaskData(copy=False),
            )
        )
    return True


def testEmptyPlot(qapp, qWidgetFactory):
    """Empty plot, display MaskToolsDockWidget, toggle multiple masks"""
    plot = qWidgetFactory(PlotWindow)

    maskDock = MaskToolsWidget.MaskToolsDockWidget(plot=plot, name="TEST")
    plot.addDockWidget(qt.Qt.BottomDockWidgetArea, maskDock)
    maskWidget = maskDock.widget()

    maskWidget.setMultipleMasks("single")
    qapp.processEvents()

    maskWidget.setMultipleMasks("exclusive")
    qapp.processEvents()


@pytest.mark.parametrize("itemMaskUpdated", [False, True])
@pytest.mark.parametrize("origin,scale", [
    ((0, 0), (1, 1)),
    ((1000, 1000), (1, 1)),
    ((0, 0), (-1, -1)),
    ((1000, 1000), (-1, -1)),
])
def testWithAnImage(qapp, qapp_utils, qWidgetFactory, origin, scale, itemMaskUpdated):
    """Plot with an image: test MaskToolsWidget interactions"""
    plot = qWidgetFactory(PlotWindow)

    maskDock = MaskToolsWidget.MaskToolsDockWidget(plot=plot, name="TEST")
    plot.addDockWidget(qt.Qt.BottomDockWidgetArea, maskDock)
    maskWidget = maskDock.widget()

    # Add and remove a image (this should enable/disable GUI + change mask)
    plot.addImage(
        numpy.random.random(1024**2).reshape(1024, 1024), legend="test"
    )
    qapp.processEvents()

    plot.remove("test", kind="image")
    qapp.processEvents()

    maskWidget.setItemMaskUpdated(itemMaskUpdated)
    plot.addImage(
        numpy.arange(1024**2).reshape(1024, 1024),
        legend="test",
        origin=origin,
        scale=scale,
    )
    qapp.processEvents()

    assert maskWidget.isItemMaskUpdated() == itemMaskUpdated

    # Test draw rectangle #
    toolButton = getQToolButtonFromAction(maskWidget.rectAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

    # mask
    maskWidget.maskStateGroup.button(1).click()
    qapp.processEvents()
    _drag(qapp, qapp_utils, plot)
    assert not numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0))
    assert _isMaskItemSync(maskWidget, plot)

    # unmask same region
    maskWidget.maskStateGroup.button(0).click()
    qapp.processEvents()
    _drag(qapp, qapp_utils, plot)
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
    assert _isMaskItemSync(maskWidget, plot)

    # Test draw polygon #
    toolButton = getQToolButtonFromAction(maskWidget.polygonAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

    # mask
    maskWidget.maskStateGroup.button(1).click()
    qapp.processEvents()
    _drawPolygon(qapp, qapp_utils, plot)
    assert not numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
    assert _isMaskItemSync(maskWidget, plot)

    # unmask same region
    maskWidget.maskStateGroup.button(0).click()
    qapp.processEvents()
    _drawPolygon(qapp, qapp_utils, plot)
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
    assert _isMaskItemSync(maskWidget, plot)

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
    assert _isMaskItemSync(maskWidget, plot)

    # unmask same region
    maskWidget.maskStateGroup.button(0).click()
    qapp.processEvents()
    _drawPencil(qapp, qapp_utils, plot)
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
    assert _isMaskItemSync(maskWidget, plot)

    # Test no draw tool #
    toolButton = getQToolButtonFromAction(maskWidget.browseAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

    plot.clear()


@pytest.mark.parametrize("file_format", ["npy", "msk"])
def testloadSave(file_format, tmp_path, qapp, qapp_utils, qWidgetFactory):
    """Plot with an image: test MaskToolsWidget operations"""
    plot = qWidgetFactory(PlotWindow)

    maskDock = MaskToolsWidget.MaskToolsDockWidget(plot=plot, name="TEST")
    plot.addDockWidget(qt.Qt.BottomDockWidgetArea, maskDock)
    maskWidget = maskDock.widget()

    plot.addImage(numpy.arange(1024**2).reshape(1024, 1024), legend="test")
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

    maskDock = MaskToolsWidget.MaskToolsDockWidget(plot=plot, name="TEST")
    plot.addDockWidget(qt.Qt.BottomDockWidgetArea, maskDock)
    maskWidget = maskDock.widget()

    plot.addImage(numpy.arange(512**2).reshape(512, 512), legend="test")
    plot.resetZoom()
    qapp.processEvents()

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
