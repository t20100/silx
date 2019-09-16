# coding: utf-8
# /*##########################################################################
# Copyright (C) 2019 European Synchrotron Radiation Facility
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
# ############################################################################*/
__authors__ = ["T. Caswell", "T. Vincent"]
__license__ = "MIT"
__date__ = "16/09/2019"

import weakref
import numpy

from . import commonh5
from .url import DataUrl
from databroker._drivers import jsonl


class _HeaderGroup(commonh5.Group):
    def __init__(self, entry, **kwargs):
        self._entry = entry
        start_doc = self._entry.metadata['start']
        super(_HeaderGroup, self).__init__("%s: %04d" % (start_doc['plan_name'], start_doc['scan_id']), **kwargs)

        for key in self._entry:
            group = commonh5.Group(key, parent=self)
            self.add_node(group)

            xary = self._entry[key].read()
            for label in xary.keys():
                if label == 'uid':
                    continue
                group.add_node(
                    commonh5.Dataset(label,
                                     data=numpy.asanyarray(xary[label])))


class DataBrokerFile(commonh5.File):

    @property
    def uid(self):
        return self.filename

    def __init__(self, name=None):
        url = DataUrl(name)
        if url.scheme() != 'broker':
            raise ValueError("Not a broker URL: %s" % name)

        server_name = url.file_path()[1:] #TODO fix use of abs path
        commonh5.File.__init__(self, name=server_name, mode="r")
        #self._broker = getarrt(cat, url.file_path())
        self._broker = jsonl.BlueskyJSONLCatalog(
            "/tmp/db_jsonl/*.jsonl",
            name=server_name)

        if url.data_path() is None:
            entries = list(self._broker)
        else:
            entries = [url.data_path()]

        for entry in entries:
            entry_group = _HeaderGroup(self._broker[entry](), parent=self)
            self.add_node(entry_group)


_files = {}


def dataBrokerFile(name):
    url = DataUrl(name)
    if url.scheme() != 'broker':
        raise ValueError("Not a broker URL: %s" % name)

    filename = url.file_path()[1:]
    if filename in _files:
        f = _files[filename]()
        if f is not None:
            return f

    f = DataBrokerFile(filename)
    _files[filename] = weakref.ref(f)
    return f
