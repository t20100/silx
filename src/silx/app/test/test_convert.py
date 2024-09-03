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
"""Module testing silx.app.convert"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/01/2018"


import io
import gc
import h5py

import silx
from .. import convert
from silx.utils import testutils
from silx.io.utils import h5py_read_dataset


# content of a spec file
sftext = """#F /tmp/sf.dat
#E 1455180875
#D Thu Feb 11 09:54:35 2016
#C imaging  User = opid17
#O0 Pslit HGap  MRTSlit UP  MRTSlit DOWN
#O1 Sslit1 VOff  Sslit1 HOff  Sslit1 VGap
#o0 pshg mrtu mrtd
#o2 ss1vo ss1ho ss1vg

#J0 Seconds  IA  ion.mono  Current
#J1 xbpmc2  idgap1  Inorm

#S 1  ascan  ss1vo -4.55687 -0.556875  40 0.2
#D Thu Feb 11 09:55:20 2016
#T 0.2  (Seconds)
#P0 180.005 -0.66875 0.87125
#P1 14.74255 16.197579 12.238283
#N 4
#L MRTSlit UP  second column  3rd_col
-1.23 5.89  8
8.478100E+01  5 1.56
3.14 2.73 -3.14
1.2 2.3 3.4

#S 1 aaaaaa
#D Thu Feb 11 10:00:32 2016
#@MCADEV 1
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
#N 3
#L uno  duo
1 2
@A 0 1 2
@A 10 9 8
3 4
@A 3.1 4 5
@A 7 6 5
5 6
@A 6 7.7 8
@A 4 3 2
"""

def testConvertHelp():
    """option -h must cause a `raise SystemExit` or a `return 0`"""
    try:
        result = convert.main(["convert", "--help"])
    except SystemExit as e:
        result = e.args[0]
    assert result == 0


def testConvertWrongOption():
    """presence of a wrong option must cause a SystemExit or a return
    with a non-zero status
    """
    try:
        result = convert.main(["convert", "--foo"])
    except SystemExit as e:
        result = e.args[0]
    assert result != 0


@testutils.validate_logging(convert._logger.name, error=3)
def testWrongFiles():
    """one error log per missing file + one "Aborted" error log"""
    result = convert.main(["convert", "foo.spec", "bar.edf"])
    assert result != 0


def testFile(tmp_path):
    # write a temporary SPEC file
    specpath = tmp_path / "input.dat"
    with io.open(specpath, "wb") as fd:
        fd.write(bytes(sftext, "ascii"))

    # convert it
    h5path = tmp_path / "output.h5"
    assert not h5path.is_file()
    command_list = ["convert", "-m", "w", str(specpath), "-o", h5path]
    result = convert.main(command_list)

    assert result == 0
    assert h5path.is_file()

    with h5py.File(h5path, "r") as h5f:
        title12 = h5py_read_dataset(h5f["/1.2/title"])
        assert title12 == "aaaaaa"

        creator = h5f.attrs.get("creator")
        assert creator is not None, "No creator attribute in NXroot group"
        assert f"silx convert (v{silx.version})" in creator

    gc.collect()  # necessary to free spec file on Windows
