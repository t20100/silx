# /*##########################################################################
#
# Copyright (c) 2017-2024 European Synchrotron Radiation Facility
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
"""Test the plot's save action (consistency of output)"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "28/11/2017"


from silx.gui.plot import PlotWidget
from silx.gui.plot.actions.io import SaveAction


def testSaveMultipleCurvesAsSpec(tmp_path, qWidgetFactory):
    """Test that labels are properly used."""
    plot = qWidgetFactory(PlotWidget, backend="none")
    saveAction = SaveAction(plot=plot)

    outPath = tmp_path / "out.dat"

    plot.setGraphXLabel("graph x label")
    plot.setGraphYLabel("graph y label")

    plot.addCurve(
        [0, 1], [1, 2], "curve with labels", xlabel="curve0 X", ylabel="curve0 Y"
    )
    plot.addCurve([-1, 3], [-6, 2], "curve with X label", xlabel="curve1 X")
    plot.addCurve([-2, 0], [8, 12], "curve with Y label", ylabel="curve2 Y")
    plot.addCurve([3, 1], [7, 6], "curve with no labels")

    saveAction._saveCurves(
        plot, str(outPath), SaveAction.DEFAULT_ALL_CURVES_FILTERS[0]
    )  # "All curves as SpecFile (*.dat)"

    with open(outPath, "rb") as f:
        file_content = f.read()
        if hasattr(file_content, "decode"):
            file_content = file_content.decode()

        # case with all curve labels specified
        assert "#S 1 curve0 Y" in file_content
        assert "#L curve0 X  curve0 Y" in file_content

        # graph X&Y labels are used when no curve label is specified
        assert "#S 2 graph y label" in file_content
        assert "#L curve1 X  graph y label" in file_content

        assert "#S 3 curve2 Y" in file_content
        assert "#L graph x label  curve2 Y" in file_content

        assert "#S 4 graph y label" in file_content
        assert "#L graph x label  graph y label" in file_content


def testFileFilterAPI(qWidgetFactory):
    """Test addition/update of a file filter"""
    plot = qWidgetFactory(PlotWidget)
    saveAction = SaveAction(plot=plot, parent=plot)

    def dummySaveFunction(plot, filename, nameFilter):
        pass

    # Add a new file filter
    nameFilter = "Dummy file (*.dummy)"
    saveAction.setFileFilter("all", nameFilter, dummySaveFunction)
    assert nameFilter in saveAction.getFileFilters("all")
    assert saveAction.getFileFilters("all")[nameFilter] is dummySaveFunction

    # Add a new file filter at a particular position
    nameFilter = "Dummy file2 (*.dummy)"
    saveAction.setFileFilter("all", nameFilter, dummySaveFunction, index=3)
    assert nameFilter in saveAction.getFileFilters("all")
    filters = saveAction.getFileFilters("all")
    assert filters[nameFilter] is dummySaveFunction
    assert list(filters.keys()).index(nameFilter) == 3

    # Update an existing file filter
    nameFilter = SaveAction.IMAGE_FILTER_EDF
    saveAction.setFileFilter("image", nameFilter, dummySaveFunction)
    assert saveAction.getFileFilters("image")[nameFilter] is dummySaveFunction

    # Change the position of an existing file filter
    nameFilter = "Dummy file2 (*.dummy)"
    oldIndex = list(saveAction.getFileFilters("all")).index(nameFilter)
    newIndex = oldIndex - 1
    saveAction.setFileFilter(
        "all", nameFilter, dummySaveFunction, index=newIndex
    )
    filters = saveAction.getFileFilters("all")
    assert filters[nameFilter] is dummySaveFunction
    assert list(filters.keys()).index(nameFilter) == newIndex
