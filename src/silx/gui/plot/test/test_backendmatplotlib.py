"""Tests of the matplotlib backend with a layout engine"""

import numpy
import pytest
from matplotlib.layout_engine import TightLayoutEngine

import silx
from silx.gui.plot import PlotWidget
from silx.gui.plot.PlotInteraction import Pan
from silx.gui.utils.matplotlib import FigureCanvasQTAgg


def _getAspectRatio(plot):
    width, height = plot.getPlotBoundsInPixels()[2:]
    xmin, xmax = plot.getXAxis().getLimits()
    ymin, ymax = plot.getYAxis().getLimits()
    return ((xmax - xmin) / width) / ((ymax - ymin) / height)


@pytest.fixture
def tightPlot(qWidgetFactory, qapp, monkeypatch):
    monkeypatch.setattr(silx.config, "_MPL_TIGHT_LAYOUT", True)
    plot = qWidgetFactory(PlotWidget, backend="mpl")
    plot.resize(600, 400)
    plot.setKeepDataAspectRatio(True)
    plot.setGraphXLabel("x")
    plot.setGraphYLabel("a long\ny label")
    plot.addImage(numpy.random.rand(100, 300) * 1e5)
    qapp.processEvents()
    plot.getWidgetHandle().draw()
    return plot


def testKeepAspectRatioWithTightLayout(tightPlot):
    """Aspect ratio is kept once the layout engine has resized the axes"""
    backend = tightPlot._backend
    assert backend.fig.get_layout_engine() is not None
    assert _getAspectRatio(tightPlot) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    "size", [(400, 300), (300, 600), (900, 300), (800, 500), (200, 200)]
)
def testKeepAspectRatioWithTightLayoutAfterResize(tightPlot, qapp, size):
    """Aspect ratio is kept after resizing the widget"""
    tightPlot.resize(*size)
    qapp.processEvents()
    tightPlot.getWidgetHandle().draw()
    assert _getAspectRatio(tightPlot) == pytest.approx(1.0, abs=1e-3)


def testKeepAspectRatioWithTightLayoutAfterPan(tightPlot, qapp):
    """Aspect ratio is kept after panning"""
    pan = Pan(tightPlot)
    pan.beginDrag(300, 200, "left")
    for i in range(1, 30):
        pan.drag(300 + i * 5, 200 + i * 3, "left")
    pan.endDrag((300, 200), (450, 290), "left")
    qapp.processEvents()
    tightPlot.getWidgetHandle().draw()

    assert _getAspectRatio(tightPlot) == pytest.approx(1.0, abs=1e-3)


def testSingleDrawWithTightLayout(tightPlot, qapp, monkeypatch):
    """Layout is solved off-screen: the figure is drawn only once"""
    backend = tightPlot._backend
    tightPlot.resize(800, 300)
    qapp.processEvents()

    nbDraws = 0
    canvasDraw = FigureCanvasQTAgg.draw

    def countingDraw(canvas):
        nonlocal nbDraws
        nbDraws += 1
        canvasDraw(canvas)

    monkeypatch.setattr(FigureCanvasQTAgg, "draw", countingDraw)
    tightPlot.resize(300, 600)
    backend.draw()

    assert nbDraws == 1
    assert _getAspectRatio(tightPlot) == pytest.approx(1.0, abs=1e-3)
    # The layout engine is restored after draw
    assert isinstance(backend.fig.get_layout_engine(), TightLayoutEngine)


# Keeping aspect ratio can lead to non-positive limits with log scale
@pytest.mark.filterwarnings("ignore:Attempt to set non-positive:UserWarning")
def testKeepAspectRatioWithTightLayoutLogScale(tightPlot, qapp):
    """Aspect ratio is kept and labels fit in the figure with log scale"""
    tightPlot.getYAxis().setScale("log")
    qapp.processEvents()
    backend = tightPlot._backend
    backend.draw()

    assert _getAspectRatio(tightPlot) == pytest.approx(1.0, abs=1e-3)

    renderer = backend.fig.canvas.get_renderer()
    figBBox = backend.fig.bbox
    axesBBox = backend.ax.get_tightbbox(renderer)
    assert axesBBox.x0 >= figBBox.x0 and axesBBox.x1 <= figBBox.x1
    assert axesBBox.y0 >= figBBox.y0 and axesBBox.y1 <= figBBox.y1
