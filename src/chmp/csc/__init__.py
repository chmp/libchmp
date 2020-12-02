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
        return [cell.name for cell in self._parse_script()]

    def get(self, cell: Union[int, str]) -> List[str]:
        cell = self._find_cell(self._parse_script(), cell)
        return cell.source.splitlines()

    def load(self, cell: Union[int, str]):
        from IPython import get_ipython

        source = self.get(cell)
        get_ipython().set_next_input("\n".join(source), replace=True)

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
        cell = self._find_cell(parsed_cells, cell)

        if self.verbose:
            self._print_cell(cell.source)

        # include leading new-lines to ensure the line offset of the source
        # matches the file. This is required fo inspect.getsource to work
        # correctly, which in turn is used for example py torch.jit.script
        source = "\n" * cell.range[0] + cell.source

        code = compile(source, str(self.path.resolve()), "exec")
        exec(code, vars(self.ns), vars(self.ns))

    def _find_cell(self, parsed_cells, cell):
        cands = [c for c in parsed_cells if c.matches(cell)]

        if len(cands) == 0:
            raise ValueError("Could not find cell")

        elif len(cands) > 1:
            raise ValueError(
                f"Found multiple cells: {', '.join(str(c.name) for c in cands)}"
            )

        return cands[0]

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
    current_cell_start = 0

    for idx, line in enumerate(fobj):
        m = cell_pattern.match(line)

        if m is None:
            current_cell_lines.append(line)

        else:
            if current_cell_name is not None or current_cell_lines:
                cell = Cell(
                    current_cell_name,
                    len(cells),
                    (current_cell_start, idx + 1),
                    "".join(current_cell_lines),
                )
                cells.append(cell)

            current_cell_start = idx + 1
            current_cell_name = m.group(1).strip()
            current_cell_lines = []

    # NOTE if current_cell_name is not None or there are lines then idx is defined
    if current_cell_name is not None or current_cell_lines:
        cell = Cell(
            current_cell_name,
            len(cells),
            (current_cell_start, idx + 1),
            "".join(current_cell_lines),
        )
        cells.append(cell)

    return cells


class Cell:
    def __init__(self, name, idx, range, source):
        self.name = name
        self.idx = idx
        self.range = range
        self.source = source

    def matches(self, cell):
        if isinstance(cell, str):
            return self.name is not None and self.name.startswith(cell.strip())

        else:
            return self.idx == cell
