import json
import pathlib
import re
import sys
import types

from typing import Any, Dict, List, Tuple, Union, Optional


class CellScript:
    """Allow to execute a python script step by step

    ``CellScript`` is designed to be used inside Jupyter nobteooks and allows to
    execute an external script with special cell annotations cell by cell. The
    script is reloaded before execution, but the namespace is persisted on this
    ``CellScript`` instance.

    Example::

        train_script = CellScript("model_training.py")

        train_script("Setup")
        train_script("Train")
        train_script("Save")

    The namespace of the script is available via the ``ns`` attribute::

        train_script("Setup")
        print("parameters:", sorted(train_script.ns.model.state_dict()))

        train_script("Train")
        train_script.ns.model
    """

    path: pathlib.Path
    verbose: bool
    cell_marker: str
    ns: Any
    export_ns: Any
    exports: Dict[str, str]
    cell_pattern: re.Pattern

    def __init__(self, path, cell_marker="%%", ns=None, export_ns=None, verbose=True):
        self.path = pathlib.Path(path)
        self.verbose = verbose
        self.cell_marker = str(cell_marker)
        self.ns = self._valid_ns(ns, self.path)
        self.export_ns = self._valid_export_ns(export_ns)

        self.exports = dict()
        self.cell_pattern = re.compile(
            r"^#\s*" + re.escape(self.cell_marker) + r"(.*)$"
        )

    @staticmethod
    def _valid_ns(ns, path):
        if ns is not None:
            return ns

        ns = types.ModuleType(path.stem)
        ns.__file__ = str(path)
        return ns

    @staticmethod
    def _valid_export_ns(ns):
        if ns is not None:
            return ns

        import __main__

        return __main__

    def export(self, *names: str, **renames: str):
        self.exports.update({name: name for name in names})
        self.exports.update(renames)
        self._export()

    def list(self) -> List[Optional[str]]:
        parsed_cells = self._parse_script()
        return [cell_name for cell_name, _ in parsed_cells]

    def eval(self, expr):
        return eval(expr.strip(), vars(self.ns), vars(self.ns))

    def run(self, *cells: Union[int, str], **assignments: Any):
        self._assign(assignments)

        parsed_cells = self._parse_script()
        for idx, cell in enumerate(cells):
            if self.verbose and idx != 0:
                print(file=sys.stderr)

            self._run(parsed_cells, cell)

        self._export()

    def _assign(self, assignments: Dict[str, Any]):
        for key, value in assignments.items():
            setattr(self.ns, key, value)

    def _parse_script(self) -> List[Tuple[str, str]]:
        with self.path.open("rt") as fobj:
            return parse_script(fobj, self.cell_pattern)

    def _run(self, parsed_cells: List[Tuple[str, str]], cell: Union[int, str]):
        cell_source = self._find_cell(parsed_cells, cell)

        if self.verbose:
            self._print_cell(cell_source)

        exec(cell_source, vars(self.ns), vars(self.ns))

    def _find_cell(
        self, parsed_cells: List[Tuple[str, str]], cell: Union[int, str]
    ) -> str:
        if isinstance(cell, str):
            cell = cell.strip()
            cands = [
                (cell_name, cell_source)
                for cell_name, cell_source in parsed_cells
                if cell_name is not None and cell_name.startswith(cell)
            ]

        else:
            cands = [
                (cell_name, cell_source)
                for idx, (cell_name, cell_source) in enumerate(parsed_cells)
                if idx == cell
            ]

        if len(cands) == 0:
            raise ValueError("Could not find cell")

        elif len(cands) > 1:
            raise ValueError(
                f"Found multiple cells: {', '.join(str(cn) for cn, _ in cands)}"
            )

        ((_, cell_source),) = cands
        return cell_source

    def _export(self):
        for target, source in self.exports.items():
            if not hasattr(self.ns, source):
                continue

            setattr(self.export_ns, target, getattr(self.ns, source))

    def _print_cell(self, cell_source):
        lines = ["  " + line for line in cell_source.strip().splitlines()]

        if len(lines) < 19:
            print("\n".join(lines), file=sys.stderr)

        else:
            print(
                "\n".join(lines[:8] + ["", "[...]", ""] + lines[-8:]),
                file=sys.stderr,
            )


def parse_script(fobj, cell_pattern):
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

            current_cell_name = m.group(1).strip()
            current_cell_lines = []

    if current_cell_name is not None or current_cell_lines:
        cells.append((current_cell_name, "".join(current_cell_lines)))

    return cells
