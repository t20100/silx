#!/usr/bin/env python3
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2022 European Synchrotron Radiation Facility
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
"""Build man pages of the provided entry points"""

import logging
import os
from pathlib import Path
import stat
import subprocess
import sys
import sysconfig
import tempfile
from typing import Iterator, Tuple

import pkg_resources


logging.basicConfig()
logger = logging.getLogger(__name__)


PROJECT = 'silx'


def entry_points(name: str) -> Iterator[Tuple[str, str, str]]:
    """Iterate over entry points available on the project `name`"""
    for group in ('console_scripts', 'gui_scripts'):
        for entry_point in pkg_resources.iter_entry_points(group, name):
            yield entry_point.name, entry_point.module_name, entry_point.attrs[0]


def _write_script(target_name, lst_lines):
    """Write a script to a temporary file and return its name

    :param target_name: base of the script name
    :param lst_lines: list of lines to be written in the script
    :return: the actual filename of the script (for execution or removal)
    """
    script_fid, script_name = tempfile.mkstemp(prefix="%s_" % target_name, text=True)
    with os.fdopen(script_fid, "wt") as script:
        for line in lst_lines:
            if not line.endswith("\n"):
                line += "\n"
            script.write(line)
    # make it executable
    mode = os.stat(script_name).st_mode
    os.chmod(script_name, mode + stat.S_IEXEC)
    return script_name


def get_synopsis(module_name, env):
    """Execute a script to retrieve the synopsis for help2man

    :return: synopsis
    :rtype: single line string
    """
    script_name = None
    synopsis = None
    script = [
        "#!%s\n" % sys.executable,
        "import logging",
        "logging.basicConfig(level=logging.ERROR)",
        "import %s as app" % module_name,
        "print(app.__doc__)",
    ]
    try:
        script_name = _write_script(module_name, script)
        command_line = [sys.executable, script_name]
        p = subprocess.Popen(command_line, env=env, stdout=subprocess.PIPE)
        status = p.wait()
        if status != 0:
            logger.warning("Error while getting synopsis for module '%s'.", module_name)
        synopsis = p.stdout.read().decode("utf-8").strip()
        if synopsis == "None":
            synopsis = None
    finally:
        # clean up the script
        if script_name is not None:
            os.remove(script_name)
    return synopsis


def run_targeted_script(target_name, script_name, env, log_output=False):
    """Execute targeted script using --help and --version to help checking errors.

    help2man is not very helpful to do it for us.

    :return: True is both return code are equal to 0
    :rtype: bool
    """
    if log_output:
        extra_args = {}
    else:
        extra_args = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}

    succeeded = True
    command_line = [sys.executable, script_name, "--help"]
    if log_output:
        logger.info("See the following execution of: %s", " ".join(command_line))
    p = subprocess.Popen(command_line, env=env, **extra_args)
    status = p.wait()
    if log_output:
        logger.info("Return code: %s", status)
    succeeded = succeeded and status == 0
    command_line = [sys.executable, script_name, "--version"]
    if log_output:
        logger.info("See the following execution of: %s", " ".join(command_line))
    p = subprocess.Popen(command_line, env=env, **extra_args)
    status = p.wait()
    if log_output:
        logger.info("Return code: %s", status)
    succeeded = succeeded and status == 0
    return succeeded


def main():
    build_lib = (
        Path(__file__).parent
        / ".."
        / "build"
        / f"lib.{sysconfig.get_platform()}-{sys.version_info[0]}.{sys.version_info[1]}"
    ).resolve()
    path = sys.path
    path.insert(0, str(build_lib))

    env = dict((str(k), str(v)) for k, v in os.environ.items())
    env["PYTHONPATH"] = os.pathsep.join(path)
    if not os.path.isdir("build/man"):
        os.makedirs("build/man")
    script_name = None
    workdir = tempfile.mkdtemp()

    for target_name, module_name, function_name in entry_points(PROJECT):
        logger.info("Build man for entry-point target '%s'" % target_name)
        # help2man expect a single executable file to extract the help
        # we create it, execute it, and delete it at the end

        try:
            # create a launcher using the right python interpreter
            script_name = os.path.join(workdir, target_name)
            with open(script_name, "wt") as script:
                script.write("#!%s\n" % sys.executable)
                script.write("import %s as app\n" % module_name)
                script.write("app.%s()\n" % function_name)
            # make it executable
            mode = os.stat(script_name).st_mode
            os.chmod(script_name, mode + stat.S_IEXEC)

            # execute help2man
            man_file = "build/man/%s.1" % target_name
            command_line = ["help2man", "-N", script_name, "-o", man_file]

            synopsis = get_synopsis(module_name, env)
            if synopsis:
                command_line += ["-n", synopsis]

            p = subprocess.Popen(command_line, env=env)
            status = p.wait()
            if status != 0:
                logger.info(
                    "Error while generating man file for target '%s'.", target_name
                )
                run_targeted_script(target_name, script_name, env, True)
                raise RuntimeError(
                    "Fail to generate '%s' man documentation" % target_name
                )
        finally:
            # clean up the script
            if script_name is not None:
                os.remove(script_name)
    os.rmdir(workdir)


if __name__ == "__main__":
    main()
