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
"""Build man pages of the project's entry points"""

import logging
from pathlib import Path
import subprocess
import sys
import sysconfig
from typing import Iterator, Tuple

import pkg_resources


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT = "silx"


def get_synopsis(module_name: str) -> str:
    """Execute Python commands to retrieve the synopsis for help2man"""
    commands = (
        "import sys",
        f"sys.path = {sys.path}",  # To use the patched sys.path
        "import logging",
        "logging.basicConfig(level=logging.ERROR)",
        f"import {module_name}",
        f"print({module_name}.__doc__)",
    )
    result = subprocess.run(
        [sys.executable, "-c", "; ".join(commands)],
        capture_output=True,
    )
    if result.returncode:
        logger.warning("Error while getting synopsis for module '%s'.", module_name)
        return None
    synopsis = result.stdout.decode("utf-8").strip()
    if synopsis == "None":
        return None
    return synopsis


def entry_points(name: str) -> Iterator[Tuple[str, str, str]]:
    for group in ("console_scripts", "gui_scripts"):
        for entry_point in pkg_resources.iter_entry_points(group, name):
            yield entry_point.name, entry_point.module_name, entry_point.attrs[0]


def main(name: str, root_path: Path):
    build_lib_path = (
        root_path
        / "build"
        / f"lib.{sysconfig.get_platform()}-{sys.version_info[0]}.{sys.version_info[1]}"
    ).resolve()
    sys.path.insert(0, str(build_lib_path))

    build_man_path = root_path / "build" / "man"
    build_man_path.mkdir(parents=True, exist_ok=True)

    for target_name, module_name, function_name in entry_points(name):
        logger.info(f"Build man for entry-point target '{target_name}'")
        commands = (
            "import sys",
            f"sys.path = {sys.path}",  # To use the patched sys.path
            f"import {module_name}",
            f"{module_name}.{function_name}()",
        )
        python_command = [sys.executable, "-c", f'"{"; ".join(commands)}"']

        help2man_command = [
            "help2man",
            "-N",
            " ".join(python_command),
            "-o",
            str(build_man_path / f"{target_name}.1"),
        ]

        synopsis = get_synopsis(module_name)
        if synopsis:
            help2man_command += ["-n", synopsis]

        result = subprocess.run(help2man_command)
        if result.returncode != 0:
            logger.error(f"Error while generating man file for target '{target_name}'.")
            for argument in ("--help", "--version"):
                test_command = python_command + [argument]
                logger.info(f"Running: {test_command}")
                result = subprocess.run(test_command)
                logger.info(f"\tReturn code: {result.returncode}")
            raise RuntimeError(f"Fail to generate '{target_name}' man documentation")


if __name__ == "__main__":
    main(PROJECT, Path(__file__).parent / "..")
