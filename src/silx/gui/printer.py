# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This module provides a singleton QPrinter used by default by silx widgets."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/03/2018"


from . import qt


_printer = None
"""Shared QPrinter instance"""


def getDefaultPrinter():
    """Returns the default printer.

    This allows reusing the same QPrinter across widgets.

    :return: QPrinter
    """
    global _printer
    if _printer is None:
        _printer = qt.QPrinter()
    return _printer


def setDefaultPrinter(printer):
    """Set the printer used by default by silx widgets.

    :param QPrinter printer:
    """
    assert isinstance(printer, qt.QPrinter)
    global _printer
    _printer = printer
