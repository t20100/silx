# /*##########################################################################
#
# Copyright (c) 2023 European Synchrotron Radiation Facility
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
"""This module provides a Text item for :class:`PlotWidget`."""

from __future__ import annotations

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "27/10/2023"


import logging

from .core import (Item, ColorMixIn, ItemChangedType, TextMixIn, YAxisMixIn)
from silx import config
from silx.gui import qt

_logger = logging.getLogger(__name__)


class Text(Item, ColorMixIn, TextMixIn, YAxisMixIn):
    """Item to display text in the plot"""

    def __init__(self):
        Item.__init__(self)
        ColorMixIn.__init__(self)
        TextMixIn.__init__(self)
        YAxisMixIn.__init__(self)

        self.__position = 0., 0.

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        if not self.getText():
            return None

        x, y = self.getPosition()
        return backend.addText(
            x=x,
            y=y,
            text=self.getText(),
            color=self.getColor(),
            yaxis=self.getYAxis(),
            font=self.getFont(copy=False),
        )

    def isOverlay(self) -> bool:
        """Returns True: A text is always rendered as an overlay"""
        return True

    def getPosition(self) -> tuple[float, float]:
        """Returns the (x, y) position of the text anchor in data coordinates"""
        return self.__position

    def setPosition(self, x: float, y: float):
        """Set text anchor position in data coordinates

        :param x: X coordinates in data frame
        :param y: Y coordinates in data frame
        """
        position = float(x), float(y)
        if self.__position != position:
            self.__position = position
            self._updated(ItemChangedType.POSITION)
