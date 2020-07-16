# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
"""Mix-in class for scene primitives"""


__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/07/2020"

import numpy

from ... import _glutils
from ..._glutils import gl


class DataTextureMixIn:
    """Mix-in class for textured scene primitive which stores the data.

    :param int ndim: Number of dimensions of the texture (2 or 3).
    :param numpy.ndarray data: Data to use to initialize the texture.
    :param bool copy:
        True (default) to copy data, False to use as is (do not modify!)
    :param internalFormat: OpenGL texture internal format
    :param format_: Input data format
    """

    def __init__(self, ndim, data, copy=True, internal=gl.GL_R32F, format_=gl.GL_RED):
        assert ndim in (2, 3)
        self.__ndim = ndim

        self.__data = None  # Initialized in __initTexture
        self.__textureToDiscard = None  # Cache texture to discard
        self.__texture = None
        self.__updateData(data, copy, internal, format_)

    @property
    def texture(self):
        """:class:`Texture` associated to this primitive."""
        return self.__texture

    def __updateData(self, data, copy, internal, format_):
        """Set the data used as texture.

        :param numpy.ndarray data:
        :param bool copy:
            True to copy data, False to use as is (do not modify!)
        :param internal: OpenGL texture internal format
        :param format_: Input data format
        """
        data = numpy.array(data, copy=copy, order='C')
        assert data.ndim == self.__ndim + (0 if format_ == gl.GL_RED else 1)
        self.__data = data

        if self.__texture is None:  # First call during __init__
            filter_ = gl.GL_LINEAR
        else:
            filter_ = self.__texture.magFilter

            if self.__texture.name is not None:
                self.__textureToDiscard = self.__texture

        self.__texture = _glutils.Texture(
            internalFormat=internal,
            data=self.__data,
            format_=format_,
            minFilter=filter_,
            magFilter=filter_,
            wrap=gl.GL_CLAMP_TO_EDGE)

    def setData(self, data, copy=True, internal=gl.GL_R32F, format_=gl.GL_RED):
        """Set the data used as texture.

        :param numpy.ndarray data:
        :param bool copy:
            True (default) to copy data, False to use as is (do not modify!)
        :param internal: OpenGL texture internal format (default: GL_R32F)
        :param format_: Input data format (default: gl.GL_RED)
        """
        self.__updateData(data, copy, internal, format_)
        self.notify()

    def getData(self, copy=True):
        """Returns the data used as the texture.

        :rtype: numpy.ndarray
        """
        return numpy.array(self.__data, copy=copy)

    @property
    def interpolation(self):
        """The texture interpolation mode: 'linear' or 'nearest'"""
        return 'nearest' if self.texture.magFilter == gl.GL_NEAREST else 'linear'

    @interpolation.setter
    def interpolation(self, interpolation):
        assert interpolation in ('linear', 'nearest')
        filter_ = gl.GL_NEAREST if interpolation == 'nearest' else gl.GL_LINEAR
        if self.texture.magFilter != filter_:
            self.texture.minFilter = filter_
            self.texture.magFilter = filter_
            self.notify()

    def prepareGL2(self, ctx):
        """Handle texture synchronization. Must be called before rendering."""
        if self.__textureToDiscard is not None:
            self.__textureToDiscard.discard()
            self.__textureToDiscard = None

        if self.texture is None:
            raise RuntimeError("Texture already discarded.")
        self.texture.prepare()

    def discard(self):
        """Release OpenGL resources"""
        if self.__textureToDiscard is not None:
            self.__textureToDiscard.discard()
            self.__textureToDiscard = None

        if self.texture is not None:
            self.texture.discard()
            self.__texture = None
