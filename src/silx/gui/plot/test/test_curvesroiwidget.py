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
"""Basic tests for CurvesROIWidget"""

__authors__ = ["T. Vincent", "P. Knobel", "H. Payno"]
__license__ = "MIT"
__date__ = "16/11/2017"


import numpy

try:
    from numpy import trapezoid
except ImportError:  # numpy v1 compatibility
    from numpy import trapz as trapezoid

from silx.gui import qt
from silx.gui.plot import items
from silx.gui.plot import Plot1D
from silx.gui.utils.testutils import TestCaseQt, SignalListener
from silx.gui.plot import PlotWindow, CurvesROIWidget
from silx.gui.utils.testutils import getQToolButtonFromAction


def testGetSetRoisAPI(qapp_utils, qWidgetFactory):
    """Simple test of the getRois and setRois API"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)
    
    roi_neg = CurvesROIWidget.ROI(
        name="negative", fromdata=-20, todata=-10, type_="X"
    )
    roi_pos = CurvesROIWidget.ROI(
        name="positive", fromdata=10, todata=20, type_="X"
    )

    widget.roiWidget.setRois((roi_pos, roi_neg))

    rois_defs = widget.roiWidget.getRois()
    widget.roiWidget.setRois(rois=rois_defs)


def testWithCurves(tmp_path, qapp_utils, qWidgetFactory):
    """Plot with curves: test all ROI widget buttons"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)

    for offset in range(2):
        plot.addCurve(
            numpy.arange(1000),
            offset + numpy.random.random(1000),
            legend=str(offset),
        )

    # Add two ROI
    qapp_utils.mouseClick(widget.roiWidget.addButton, qt.Qt.LeftButton)
    qapp_utils.qWait(200)
    qapp_utils.mouseClick(widget.roiWidget.addButton, qt.Qt.LeftButton)
    qapp_utils.qWait(200)

    # Change active curve
    plot.setActiveCurve(str(1))

    # Delete a ROI
    qapp_utils.mouseClick(widget.roiWidget.delButton, qt.Qt.LeftButton)
    qapp_utils.qWait(200)

    tmpFilePath = tmp_path / "test.ini"

    # Save ROIs
    widget.roiWidget.save(str(tmpFilePath))
    assert tmpFilePath.is_file()
    assert len(widget.getRois()) == 2

    # Reset ROIs
    qapp_utils.mouseClick(widget.roiWidget.resetButton, qt.Qt.LeftButton)
    qapp_utils.qWait(200)
    rois = widget.getRois()
    assert len(rois) == 1
    roiID = list(rois.keys())[0]
    assert rois[roiID].getName() == "ICR"

    # Load ROIs
    widget.roiWidget.load(tmpFilePath)
    assert len(widget.getRois()) == 2


def testMiddleMarker(qapp_utils, qWidgetFactory):
    """Test with middle marker enabled"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)

    widget.roiWidget.roiTable.setMiddleROIMarkerFlag(True)

    # Add a ROI
    qapp_utils.mouseClick(widget.roiWidget.addButton, qt.Qt.LeftButton)

    for roiID in widget.roiWidget.roiTable._markersHandler._roiMarkerHandlers:
        handler = widget.roiWidget.roiTable._markersHandler._roiMarkerHandlers[
            roiID
        ]
        assert handler.getMarker("min")
        xleftMarker = handler.getMarker("min").getXPosition()
        xMiddleMarker = handler.getMarker("middle").getXPosition()
        xRightMarker = handler.getMarker("max").getXPosition()
        thValue = xleftMarker + (xRightMarker - xleftMarker) / 2.0
        assert numpy.allclose(xMiddleMarker, thValue)


def testAreaCalculation(qapp_utils, qWidgetFactory):
    """Test result of area calculation"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)

    x = numpy.arange(100.0)
    y = numpy.arange(100.0)

    # Add two curves
    plot.addCurve(x, y, legend="positive")
    plot.addCurve(-x, y, legend="negative")

    # Make sure there is an active curve and it is the positive one
    plot.setActiveCurve("positive")

    # Add two ROIs
    roi_neg = CurvesROIWidget.ROI(
        name="negative", fromdata=-20, todata=-10, type_="X"
    )
    roi_pos = CurvesROIWidget.ROI(
        name="positive", fromdata=10, todata=20, type_="X"
    )

    widget.roiWidget.setRois((roi_pos, roi_neg))

    posCurve = plot.getCurve("positive")
    negCurve = plot.getCurve("negative")

    assert roi_pos.computeRawAndNetArea(posCurve) == (trapezoid(y=[10, 20], x=[10, 20]), 0.0)
    assert roi_pos.computeRawAndNetArea(negCurve) == (0.0, 0.0)
    assert roi_neg.computeRawAndNetArea(posCurve) == ((0.0), 0.0)
    assert roi_neg.computeRawAndNetArea(negCurve) == ((-150.0), 0.0)


def testCountsCalculation(qapp_utils, qWidgetFactory):
    """Test result of count calculation"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)

    x = numpy.arange(100.0)
    y = numpy.arange(100.0)

    # Add two curves
    plot.addCurve(x, y, legend="positive")
    plot.addCurve(-x, y, legend="negative")

    # Make sure there is an active curve and it is the positive one
    plot.setActiveCurve("positive")

    # Add two ROIs
    roi_neg = CurvesROIWidget.ROI(
        name="negative", fromdata=-20, todata=-10, type_="X"
    )
    roi_pos = CurvesROIWidget.ROI(
        name="positive", fromdata=10, todata=20, type_="X"
    )

    widget.roiWidget.setRois((roi_pos, roi_neg))

    posCurve = plot.getCurve("positive")
    negCurve = plot.getCurve("negative")

    assert roi_pos.computeRawAndNetCounts(posCurve) == (y[10:21].sum(), 0.0)
    assert roi_pos.computeRawAndNetCounts(negCurve) == (0.0, 0.0)
    assert roi_neg.computeRawAndNetCounts(posCurve) == ((0.0), 0.0)
    assert roi_neg.computeRawAndNetCounts(negCurve) == (y[10:21].sum(), 0.0)


def testDeferedInit(qapp_utils, qWidgetFactory):
    """Test behavior of the deferedInit"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)

    x = numpy.arange(100.0)
    y = numpy.arange(100.0)
    plot.addCurve(x=x, y=y, legend="name", replace="True")
    roisDefs = dict(
        [
            ["range1", dict([["from", 20], ["to", 200], ["type", "energy"]])],
            ["range2", dict([["from", 300], ["to", 500], ["type", "energy"]])],
        ]
    )

    plot.getCurvesRoiDockWidget().setRois(roisDefs)
    assert len(widget.roiWidget.getRois()) == len(roisDefs)
    plot.getCurvesRoiDockWidget().setVisible(True)
    assert len(widget.roiWidget.getRois()) == len(roisDefs)


def testDictCompatibility():
    """Test that ROI api is valid with dict and not information is lost"""
    roiDict = {
        "from": 20,
        "to": 200,
        "type": "energy",
        "comment": "no",
        "name": "myROI",
        "calibration": [1, 2, 3],
    }
    roi = CurvesROIWidget.ROI._fromDict(roiDict)
    assert roi.toDict() == roiDict


def testShowAllROI(qapp, qapp_utils, qWidgetFactory):
    """Test the show allROI action"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)

    x = numpy.arange(100.0)
    y = numpy.arange(100.0)
    plot.addCurve(x=x, y=y, legend="name", replace="True")

    roisDefsDict = {
        "range1": {"from": 20, "to": 200, "type": "energy"},
        "range2": {"from": 300, "to": 500, "type": "energy"},
    }

    roisDefsObj = (
        CurvesROIWidget.ROI(name="range3", fromdata=20, todata=200, type_="energy"),
        CurvesROIWidget.ROI(
            name="range4", fromdata=300, todata=500, type_="energy"
        ),
    )
    widget.roiWidget.showAllMarkers(True)
    roiWidget = plot.getCurvesRoiDockWidget().roiWidget
    roiWidget.setRois(roisDefsDict)
    markers = [
        item for item in plot.getItems() if isinstance(item, items.MarkerBase)
    ]
    assert len(markers) == 2 * 3

    markersHandler = widget.roiWidget.roiTable._markersHandler
    roiWidget.showAllMarkers(True)
    ICRROI = markersHandler.getVisibleRois()
    assert len(ICRROI) == 2

    roiWidget.showAllMarkers(False)
    ICRROI = markersHandler.getVisibleRois()
    assert len(ICRROI) == 1

    roiWidget.setRois(roisDefsObj)
    qapp.processEvents()
    markers = [
        item for item in plot.getItems() if isinstance(item, items.MarkerBase)
    ]
    assert len(markers) == 2 * 3

    markersHandler = widget.roiWidget.roiTable._markersHandler
    roiWidget.showAllMarkers(True)
    ICRROI = markersHandler.getVisibleRois()
    assert len(ICRROI) == 2

    roiWidget.showAllMarkers(False)
    ICRROI = markersHandler.getVisibleRois()
    assert len(ICRROI) == 1


def testRoiEdition(qapp_utils, qWidgetFactory):
    """Make sure if the ROI object is edited the ROITable will be updated"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)

    roi = CurvesROIWidget.ROI(name="linear", fromdata=0, todata=5)
    widget.roiWidget.setRois((roi,))

    x = (0, 1, 1, 2, 2, 3)
    y = (1, 1, 2, 2, 1, 1)
    plot.addCurve(x=x, y=y, legend="linearCurve")
    plot.setActiveCurve(legend="linearCurve")
    widget.calculateROIs()

    roiTable = widget.roiWidget.roiTable
    indexesColumns = CurvesROIWidget.ROITable.COLUMNS_INDEX
    itemRawCounts = roiTable.item(0, indexesColumns["Raw Counts"])
    itemNetCounts = roiTable.item(0, indexesColumns["Net Counts"])

    assert itemRawCounts.text() == "8.0"
    assert itemNetCounts.text() == "2.0"

    itemRawArea = roiTable.item(0, indexesColumns["Raw Area"])
    itemNetArea = roiTable.item(0, indexesColumns["Net Area"])

    assert itemRawArea.text() == "4.0"
    assert itemNetArea.text() == "1.0"

    roi.setTo(2)
    itemRawArea = roiTable.item(0, indexesColumns["Raw Area"])
    assert itemRawArea.text() == "3.0"
    roi.setFrom(1)
    itemRawArea = roiTable.item(0, indexesColumns["Raw Area"])
    assert itemRawArea.text() == "2.0"


def testRemoveActiveROI(qapp_utils, qWidgetFactory):
    """Test widget behavior when removing the active ROI"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)

    roi = CurvesROIWidget.ROI(name="linear", fromdata=0, todata=5)
    widget.roiWidget.setRois((roi,))

    widget.roiWidget.roiTable.setActiveRoi(None)
    assert len(widget.roiWidget.roiTable.selectedItems()) == 0
    widget.roiWidget.setRois((roi,))
    plot.setActiveCurve(legend="linearCurve")
    widget.calculateROIs()


def testEmitCurrentROI(qapp, qapp_utils, qWidgetFactory):
    """Test behavior of the CurvesROIWidget.sigROISignal"""
    plot = qWidgetFactory(PlotWindow)
    widget = plot.getCurvesRoiDockWidget()
    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)

    roi = CurvesROIWidget.ROI(name="linear", fromdata=0, todata=5)
    widget.roiWidget.setRois((roi,))
    signalListener = SignalListener()
    widget.roiWidget.sigROISignal.connect(signalListener.partial())
    widget.show()
    qapp.processEvents()
    assert signalListener.callCount() == 0
    assert widget.roiWidget.roiTable.activeRoi is roi
    roi.setFrom(0.0)
    qapp.processEvents()
    assert signalListener.callCount() == 0
    roi.setFrom(0.3)
    qapp.processEvents()
    assert signalListener.callCount() == 1


class TestRoiWidgetSignals(TestCaseQt):
    """Test Signals emitted by the RoiWidgetSignals"""

    def setUp(self):
        super().setUp()

        self.plot = Plot1D()
        x = range(20)
        y = range(20)
        self.plot.addCurve(x, y, legend="curve0")
        self.listener = SignalListener()
        self.curves_roi_widget = self.plot.getCurvesRoiWidget()
        self.curves_roi_widget.sigROISignal.connect(self.listener)
        assert self.curves_roi_widget.isVisible() is False
        assert self.listener.callCount() == 0
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

        toolButton = getQToolButtonFromAction(self.plot.getRoiAction())
        self.qapp.processEvents()
        self.mouseClick(widget=toolButton, button=qt.Qt.LeftButton)

        self.curves_roi_widget.show()
        self.qWaitForWindowExposed(self.curves_roi_widget)

    def tearDown(self):
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot

        self.curves_roi_widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.curves_roi_widget.close()
        del self.curves_roi_widget

        super().tearDown()

    def testSigROISignalAddRmRois(self):
        """Test SigROISignal when adding and removing ROIS"""
        self.listener.clear()

        roi1 = CurvesROIWidget.ROI(name="linear", fromdata=0, todata=5)
        self.curves_roi_widget.roiTable.addRoi(roi1)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.listener.arguments()[0][0]["current"] == "linear")
        self.listener.clear()

        roi2 = CurvesROIWidget.ROI(name="linear2", fromdata=0, todata=5)
        self.curves_roi_widget.roiTable.addRoi(roi2)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.listener.arguments()[0][0]["current"] == "linear2")
        self.listener.clear()

        self.curves_roi_widget.roiTable.removeROI(roi2)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.curves_roi_widget.roiTable.activeRoi == roi1)
        self.assertTrue(self.listener.arguments()[0][0]["current"] == "linear")
        self.listener.clear()

        self.curves_roi_widget.roiTable.deleteActiveRoi()
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.curves_roi_widget.roiTable.activeRoi is None)
        self.assertTrue(self.listener.arguments()[0][0]["current"] is None)
        self.listener.clear()

        self.curves_roi_widget.roiTable.addRoi(roi1)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.listener.arguments()[0][0]["current"] == "linear")
        self.assertTrue(self.curves_roi_widget.roiTable.activeRoi == roi1)
        self.listener.clear()
        self.qapp.processEvents()

        self.curves_roi_widget.roiTable.removeROI(roi1)
        self.qapp.processEvents()
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.listener.arguments()[0][0]["current"] == "ICR")
        self.listener.clear()

    def testSigROISignalModifyROI(self):
        """Test SigROISignal when modifying it"""
        self.curves_roi_widget.roiTable.setMiddleROIMarkerFlag(True)
        roi1 = CurvesROIWidget.ROI(name="linear", fromdata=2, todata=5)
        self.curves_roi_widget.roiTable.addRoi(roi1)
        self.curves_roi_widget.roiTable.setActiveRoi(roi1)

        # test modify the roi2 object
        self.listener.clear()
        roi1.setFrom(0.56)
        self.assertEqual(self.listener.callCount(), 1)
        self.listener.clear()
        roi1.setTo(2.56)
        self.assertEqual(self.listener.callCount(), 1)
        self.listener.clear()
        roi1.setName("linear2")
        self.assertEqual(self.listener.callCount(), 1)
        self.listener.clear()
        roi1.setType("new type")
        self.assertEqual(self.listener.callCount(), 1)

        widget = self.plot.getWidgetHandle()
        widget.setFocus(qt.Qt.OtherFocusReason)
        self.plot.raise_()
        self.qapp.processEvents()

        # modify roi limits (from the gui)
        roi_marker_handler = (
            self.curves_roi_widget.roiTable._markersHandler.getMarkerHandler(
                roi1.getID()
            )
        )
        for marker_type in ("min", "max", "middle"):
            with self.subTest(marker_type=marker_type):
                self.listener.clear()
                marker = roi_marker_handler.getMarker(marker_type)
                x_pix, y_pix = self.plot.dataToPixel(
                    marker.getXPosition(), marker.getYPosition()
                )
                self.mouseMove(widget, pos=(x_pix, y_pix))
                self.qWait(100)
                self.mousePress(widget, qt.Qt.LeftButton, pos=(x_pix, y_pix))
                self.mouseMove(widget, pos=(x_pix + 20, y_pix))
                self.qWait(100)
                self.mouseRelease(widget, qt.Qt.LeftButton, pos=(x_pix + 20, y_pix))
                self.qWait(100)
                self.mouseMove(widget, pos=(x_pix, y_pix))
                self.qapp.processEvents()
                self.assertEqual(self.listener.callCount(), 1)

    def testSetActiveCurve(self):
        """Test sigRoiSignal when set an active curve"""
        roi1 = CurvesROIWidget.ROI(name="linear", fromdata=2, todata=5)
        self.curves_roi_widget.roiTable.setActiveRoi(roi1)
        self.listener.clear()
        self.plot.setActiveCurve("curve0")
        self.assertEqual(self.listener.callCount(), 0)
