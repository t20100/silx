# /*##########################################################################
#
# Copyright (c) 2016-2023 European Synchrotron Radiation Facility
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
from __future__ import annotations

"""This module defines two main classes:

    - :class:`FrameBrowser`: a widget with 4 buttons (first, previous, next,
      last) to browse between frames and a text entry to access a specific frame
      by typing it's number)
    - :class:`HorizontalSliderWithBrowser`: a FrameBrowser with an additional
      slider. This class inherits :class:`qt.QAbstractSlider`.

"""
from silx.gui import qt
from silx.gui import icons

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "16/01/2017"


class FrameBrowser(qt.QWidget):
    """Frame browser widget, with 4 buttons/icons and a line edit to provide
    a way of selecting a frame index in a stack of images.

    .. image:: img/FrameBrowser.png

    It can be used in more generic case to select an integer within a range.

    :param QWidget parent: Parent widget
    :param int n: Number of frames. This will set the range
        of frame indices to 0--n-1.
        If None, the range is initialized to the default QSlider range (0--99).
    """

    sigIndexChanged = qt.pyqtSignal(object)

    def __init__(self, parent=None, n=None):
        qt.QWidget.__init__(self, parent)

        # Use the font size as the icon size to avoid to create bigger buttons
        fontMetric = self.fontMetrics()
        iconSize = qt.QSize(fontMetric.height(), fontMetric.height())

        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.firstButton = qt.QPushButton(self)
        self.firstButton.setIcon(icons.getQIcon("first"))
        self.firstButton.setIconSize(iconSize)
        self.previousButton = qt.QPushButton(self)
        self.previousButton.setIcon(icons.getQIcon("previous"))
        self.previousButton.setIconSize(iconSize)
        self._lineEdit = qt.QLineEdit(self)

        self._label = qt.QLabel(self)
        self.nextButton = qt.QPushButton(self)
        self.nextButton.setIcon(icons.getQIcon("next"))
        self.nextButton.setIconSize(iconSize)
        self.lastButton = qt.QPushButton(self)
        self.lastButton.setIcon(icons.getQIcon("last"))
        self.lastButton.setIconSize(iconSize)

        self.mainLayout.addWidget(self.firstButton)
        self.mainLayout.addWidget(self.previousButton)
        self.mainLayout.addWidget(self._lineEdit)
        self.mainLayout.addWidget(self._label)
        self.mainLayout.addWidget(self.nextButton)
        self.mainLayout.addWidget(self.lastButton)

        if n is None:
            first = qt.QSlider().minimum()
            last = qt.QSlider().maximum()
        else:
            first, last = 0, n

        self._lineEdit.setFixedWidth(
            self._lineEdit.fontMetrics().boundingRect("%05d" % last).width()
        )
        validator = qt.QIntValidator(first, last, self._lineEdit)
        self._lineEdit.setValidator(validator)
        self._lineEdit.setText("%d" % first)
        self._label.setText("of %d" % last)

        self._index = first
        """0-based index"""

        self.firstButton.clicked.connect(self._firstClicked)
        self.previousButton.clicked.connect(self._previousClicked)
        self.nextButton.clicked.connect(self._nextClicked)
        self.lastButton.clicked.connect(self._lastClicked)
        self._lineEdit.editingFinished.connect(self._textChangedSlot)

    def lineEdit(self):
        """Returns the line edit provided by this widget.

        :rtype: qt.QLineEdit
        """
        return self._lineEdit

    def limitWidget(self):
        """Returns the widget displaying axes limits.

        :rtype: qt.QLabel
        """
        return self._label

    def _firstClicked(self):
        """Select first/lowest frame number"""
        self.setValue(self.getRange()[0])

    def _previousClicked(self):
        """Select previous frame number"""
        self.setValue(self.getValue() - 1)

    def _nextClicked(self):
        """Select next frame number"""
        self.setValue(self.getValue() + 1)

    def _lastClicked(self):
        """Select last/highest frame number"""
        self.setValue(self.getRange()[1])

    def _textChangedSlot(self):
        """Select frame number typed in the line edit widget"""
        txt = self._lineEdit.text()
        if not len(txt):
            self._lineEdit.setText("%d" % self._index)
            return
        new_value = int(txt)
        if new_value == self._index:
            return
        ddict = {
            "event": "indexChanged",
            "old": self._index,
            "new": new_value,
            "id": id(self),
        }
        self._index = new_value
        self.sigIndexChanged.emit(ddict)

    def getRange(self):
        """Returns frame range

        :return: (first_index, last_index)
        """
        validator = self.lineEdit().validator()
        return validator.bottom(), validator.top()

    def setRange(self, first, last):
        """Set minimum and maximum frame indices.

        Initialize the frame index to *first*.
        Update the label text to *" limits: first, last"*

        :param int first: Minimum frame index
        :param int last: Maximum frame index"""
        bottom = min(first, last)
        top = max(first, last)
        self._lineEdit.validator().setTop(top)
        self._lineEdit.validator().setBottom(bottom)
        self.setValue(bottom)

        # Update limits
        self._label.setText(" limits: %d, %d " % (bottom, top))

    def setNFrames(self, nframes):
        """Set minimum=0 and maximum=nframes-1 frame numbers.

        Initialize the frame index to 0.
        Update the label text to *"1 of nframes"*

        :param int nframes: Number of frames"""
        top = nframes - 1
        self.setRange(0, top)
        # display 1-based index in label
        self._label.setText(" of %d " % top)

    def getValue(self):
        """Return current frame index"""
        return self._index

    def setValue(self, value):
        """Set 0-based frame index

        Value is clipped to current range.

        :param int value: Frame number"""
        bottom = self.lineEdit().validator().bottom()
        top = self.lineEdit().validator().top()
        value = int(value)

        if value < bottom:
            value = bottom
        elif value > top:
            value = top

        self._lineEdit.setText("%d" % value)
        self._textChangedSlot()


class _SliderPlayWidgetAction(qt.QWidgetAction):

    sigValueChanged = qt.Signal(int)

    def __init__(
        self,
        parent: qt.QWidget | None = None,
        label: str | None = None,
        tooltip: str | None = None,
    ):
        super().__init__(parent)
        self._build(label=label, tooltip=tooltip)

    def _build(self, label: str, tooltip: str):
        widget = qt.QWidget()
        layout = qt.QHBoxLayout()
        widget.setLayout(layout)
        self._spinbox = qt.QSpinBox()
        self._spinbox.setToolTip(tooltip)
        self._spinbox.setRange(1, 1000000)
        self._spinbox.valueChanged.connect(self.sigValueChanged)
        label = qt.QLabel(label)
        label.setToolTip(tooltip)
        layout.addWidget(label)
        layout.addWidget(self._spinbox)
        self.setDefaultWidget(widget)

    def value(self) -> int:
        return self._spinbox.value()

    def setValue(self, value: int):
        self._spinbox.setValue(value)


class _PlayButtonContextMenu(qt.QMenu):

    sigFrameRateChanged = qt.Signal(int)

    def __init__(self, parent: qt.QWidget | None = None):
        super().__init__(parent)
        self._build()

    def _build(self):
        self._framerateAction = _SliderPlayWidgetAction(
            self, label="FPS:", tooltip="Display speed in frames per second"
        )
        self._framerateAction.sigValueChanged.connect(self.sigFrameRateChanged)
        self._stepAction = _SliderPlayWidgetAction(
            self, label="Step:", tooltip="Step between displayed frames"
        )
        self.addAction(self._framerateAction)
        self._framerateAction.setValue(10)
        self.addAction(self._stepAction)
        self._stepAction.setValue(1)

    def getFrameRate(self) -> int:
        return self._framerateAction.value()

    def setFrameRate(self, rate: int):
        self._framerateAction.setValue(rate)

    def getStep(self) -> int:
        return self._stepAction.value()

    def setStep(self, interval: int):
        self._stepAction.setValue(interval)


class HorizontalSliderWithBrowser(qt.QAbstractSlider):
    """
    Slider widget combining a :class:`QSlider` and a :class:`FrameBrowser`.

    .. image:: img/HorizontalSliderWithBrowser.png

    The data model is an integer within a range.

    The default value is the default :class:`QSlider` value (0),
    and the default range is the default QSlider range (0 -- 99)

    The signal emitted when the value is changed is the usual QAbstractSlider
    signal :attr:`valueChanged`. The signal carries the value (as an integer).

    :param QWidget parent: Optional parent widget
    """

    def __init__(self, parent=None):
        qt.QAbstractSlider.__init__(self, parent)
        self.setOrientation(qt.Qt.Horizontal)

        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        self._slider = qt.QSlider(self)
        self._slider.setOrientation(qt.Qt.Horizontal)

        self._browser = FrameBrowser(self)

        self.mainLayout.addWidget(self._slider, 1)
        self.mainLayout.addWidget(self._browser)

        self._slider.valueChanged[int].connect(self._sliderSlot)
        self._browser.sigIndexChanged.connect(self._browserSlot)

        fontMetric = self.fontMetrics()
        iconSize = qt.QSize(fontMetric.height(), fontMetric.height())

        self.__timer = qt.QTimer(self)
        self.__timer.timeout.connect(self._updateState)

        self._playButton = qt.QToolButton(self)
        self._playButton.setToolTip("Display dataset movie.")
        self._playButton.setIcon(icons.getQIcon("camera"))
        self._playButton.setIconSize(iconSize)
        self._playButton.setCheckable(True)
        self.mainLayout.addWidget(self._playButton)

        self._playButton.toggled.connect(self._playButtonToggled)
        self._menuPlaySlider = _PlayButtonContextMenu(self)
        self._menuPlaySlider.sigFrameRateChanged.connect(self._frameRateChanged)
        self._frameRateChanged(self.getFrameRate())
        self._playButton.setMenu(self._menuPlaySlider)
        self._playButton.setPopupMode(qt.QToolButton.MenuButtonPopup)

    def lineEdit(self):
        """Returns the line edit provided by this widget.

        :rtype: qt.QLineEdit
        """
        return self._browser.lineEdit()

    def limitWidget(self):
        """Returns the widget displaying axes limits.

        :rtype: qt.QLabel
        """
        return self._browser.limitWidget()

    def setMinimum(self, value):
        """Set minimum value

        :param int value: Minimum value"""
        self._slider.setMinimum(value)
        maximum = self._slider.maximum()
        self._browser.setRange(value, maximum)

    def setMaximum(self, value):
        """Set maximum value

        :param int value: Maximum value
        """
        self._slider.setMaximum(value)
        minimum = self._slider.minimum()
        self._browser.setRange(minimum, value)

    def setRange(self, first, last):
        """Set minimum/maximum values

        :param int first: Minimum value
        :param int last: Maximum value"""
        self._slider.setRange(first, last)
        self._browser.setRange(first, last)

    def _sliderSlot(self, value):
        """Emit selected value when slider is activated"""
        self._browser.setValue(value)
        self.valueChanged.emit(value)

    def _browserSlot(self, ddict):
        """Emit selected value when browser state is changed"""
        self._slider.setValue(ddict["new"])

    def setValue(self, value):
        """Set value

        :param int value: value"""
        self._slider.setValue(value)
        self._browser.setValue(value)

    def value(self):
        """Get selected value"""
        return self._slider.value()

    def setFrameRate(self, value: int):
        """Set the frame rate at which images are displayed"""
        self._menuPlaySlider.setFrameRate(value)

    def getFrameRate(self) -> int:
        """Get the frame rate at which images are displayed"""
        return self._menuPlaySlider.getFrameRate()

    def setPlayImageStep(self, value: int):
        """Set the step between displayed images when playing"""
        self._menuPlaySlider.setStep(value)

    def getPlayImageStep(self) -> int:
        """Returns the step between displayed images"""
        return self._menuPlaySlider.getStep()

    def _frameRateChanged(self, framerate: int):
        """Update the timer interval"""
        self.__timer.setInterval(int(1 / framerate * 1e3))

    def _playButtonToggled(self, checked: bool):
        """Start/Stop the slider sequence."""
        if checked:
            self.__timer.start()
            return
        self.__timer.stop()

    def _updateState(self):
        """Advance an interval number of frames in the browser sequence."""
        currentIndex = self._browser.getValue()
        if currentIndex < self._browser.getRange()[-1]:
            self.setValue(currentIndex + self.getPlayImageStep())
        else:
            self._playButton.setChecked(False)
