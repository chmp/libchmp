import json
import re
import sys

import __main__

from IPython import get_ipython
from IPython.core.magic import Magics, magics_class, cell_magic, line_cell_magic


@magics_class
class IScriptMagics(Magics):
    @line_cell_magic("")
    def short(self, line, cell=None):
        return self._impl(line, cell, "")

    @line_cell_magic("pysync")
    def pysync(self, line, cell=None):
        return self._impl(line, cell, "pysync")

    def _impl(self, line, cell, alias):
        if not hasattr(__main__, "__pysync_file__"):
            print("Please define __pysync_file__", file=sys.stderr)
            return

        try:
            idx = int(line)

        except ValueError as exc:
            print(
                f"Could not parse cell index. Usage: %%{alias} cell_idx",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            return

        path = getattr(__main__, "__pysync_file__")
        idx = int(line)

        cell_source = read_cell_from_script(path, idx)
        cell_source.rstrip()

        if cell is not None and cell.strip() == cell_source.strip():
            self.shell.run_cell(cell)

        else:
            self.shell.set_next_input(
                f"%%{alias} {line}\n" + cell_source.rstrip(), replace=True
            )


def read_cell_from_script(path, idx):
    # TODO:optimize this call
    with open(path, "rt") as fobj:
        cells = parse_script(fobj)

    return cells[idx][1]


def parse_script(fobj):
    cells = []
    current_cell_name = None
    current_cell_lines = []

    for line in fobj:
        m = cell_pattern.match(line)

        if m is None:
            current_cell_lines.append(line)

        else:
            if current_cell_name is not None or current_cell_lines:
                cells.append((current_cell_name, "".join(current_cell_lines)))

            current_cell_name = m.group(1)
            current_cell_lines = []

    if current_cell_name is not None or current_cell_lines:
        cells.append((current_cell_name, "".join(current_cell_lines)))

    return cells


cell_pattern = re.compile(r"^#\s*%%(.*)$")
get_ipython().register_magics(IScriptMagics)
