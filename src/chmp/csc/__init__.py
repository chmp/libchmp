"""Execution of scripts section by section.

Sometimes it may be helpful to run individual parts of a script inside an
interactive environment, for example Jupyter Notebooks. ``CellScript`` is
designed to support this use case. The basis are Pythn scripts with special cell
annotations. For example consider a script to define and train a model::

    #%% Setup
    ...

    #%% Train
    ...

    #%% Save  
    ...

Where each of the ``...`` stands for arbitrary user defined code. Using
``CellScript`` this script can be step by step as::

    script = CellScript("external_script.py")

    script.run("Setup")
    script.run("Train")
    script.run("Save")

To list all available cells use ``script.list()``. 

The variables defined inside the script can be accessed and modified using the
``ns`` attribute of the script. One example would be to define a parameter cell
with default parameters and the overwrite the values before executing the
remaining cells. Assume the script defines a parameter cell as follows::

    #%% Parameters
    hidden_units = 128
    activation = 'relu'

Then the parameters can be modified as in::

    script.run("Parameters")
    script.ns.hidden_units = 64
    script.ns.activation = 'sigmoid'

Beyond direct modification of the script namespace ``CellScript`` offers 
different ways to interact with the script namespace. With 
``script.assign`` variables inside the script can be overwritten. For example,
the assignment from before could also have been written as::

    script.assign(hidden_units=64, activation='sigmoid')

Script objects can also export variables into the ``__main__`` module, which is
the namespace for example for Jupyter Notebooks. Exports can be declared with::

    script.export("loss_history", "model")

After exports are declared, the variables are copied from the script namespace 
into the export namespace after each call to ``run``.

"""
import enum
import pathlib
import re
import sys
import textwrap
import types

from typing import Any, Dict, List, Tuple, Union, Optional

__all__ = ["CellScript"]


class CellScript:
    """Allow to execute a python script step by step

    ``CellScript`` is designed to be used inside Jupyter notebooks and allows to
    execute an external script with special cell annotations cell by cell. The
    script is reloaded before execution, but the namespace is persisted on this
    ``CellScript`` instance.

    The namespace of the script is available via the ``ns`` attribute::

        train_script("Setup")
        print("parameters:", sorted(train_script.ns.model.state_dict()))

        train_script("Train")
        train_script.ns.model

    :param path:
        The path of the script, can be a string or a :class:`pathlib.Path`.
    :param cell_marker:
        The cell marker used. Cells are defined as ``# {CELL_MARKER} {NAME}``,
        with an arbitrary number of spaces allowed.
    :param verbose:
        If True, print a summary of the code executed for each cell.
    :param ns:
        The namespace to use for the execution. Per default a new module will
        be constructed. To share the same namespace with the currently running
        notebook it can be set to the ``__main__`` module.
    :param export_ns:
        The namespace to use for variable exports, see also the ``export``
        method. Per default the ``__main__`` module will be used.
    """

    path: pathlib.Path
    verbose: bool
    cell_marker: str
    ns: Any
    export_ns: Any
    exports: Dict[str, str]
    cell_pattern: re.Pattern

    def __init__(
        self,
        path,
        *,
        cell_marker="%%",
        verbose=True,
        ns=None,
        export_ns=None,
    ):
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

    def run(self, *cells: Union[int, str]):
        """Execute cells inside the script

        :param cells:
            The cells to execute. They can be passed as the index of the cell
            or its name. Cell names only have to match the beginning of the
            name, as long as the prefix uniquely defines a cell. For example,
            instead of ``"Setup training"`` also ``"Setup"`` can be used.
        """
        parsed_cells = self._parse_script()
        for idx, cell in enumerate(cells):
            if self.verbose and idx != 0:
                print(file=sys.stderr)

            self._run(parsed_cells, cell)

        self._export()

    def list(self) -> List[Optional[str]]:
        """List the names for all cells inside the script.

        If a cell is unnamed, ``None`` will be returned.
        """
        return [cell.name for cell in self._parse_script()]

    def get(self, cell: Union[int, str]) -> List[str]:
        """Get the source code of a cell

        See the ``run`` method for details of what values can be used for the
        cell parameter.
        """
        cell = self._find_cell(self._parse_script(), cell)
        return cell.source.splitlines()

    def assign(self, **assignments: Any):
        """Assign variables in the script namespace.

        :param assignments:
            values given as ``variable=value`` pairs.
        """
        for key, value in assignments.items():
            setattr(self.ns, key, value)

    def export(self, *names: str, **renames: str):
        """Declare exports

        :param names:
            variables which will be exported as-is into the export scope.
        :param renames:
            variables which will be renamed, when exported. For example, the
            declaration ``script.export(script_model="model")`` would export
            the ``model``  variable in the script namespace as ``script_model``
            into the export namespace.
        """
        self.exports.update({name: name for name in names})
        self.exports.update(renames)
        self._export()

    def eval(self, expr: str):
        """Execute an expression inside the script namespace.

        The expression can also be passed as a multiline string::

            result = script.eval('''
                a + b
            ''')

        """
        # NOTE add parens to make multiline expressions safe
        return eval("(" + expr.strip() + ")", vars(self.ns), vars(self.ns))

    def exec(self, source: str):
        """Execute a Python block inside the script namespace.

        The source is dedented to  allow using  ``.eval`` inside nested
        blocks::

            if cond:
                script.exec('''
                    hello = 'world'
                ''')

        """
        source = source.strip()
        source = textwrap.dedent(source)

        exec(source, vars(self.ns), vars(self.ns))

    def load(self, cell: Union[int, str]):
        """Load a cell into the notebook.

        This function will replace the current notebook cell with the content
        of the given script cell. The idea is to allow quick modification of
        script code inside the notebook.
        """
        from IPython import get_ipython

        source = self.get(cell)
        get_ipython().set_next_input("\n".join(source), replace=True)

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
        lines = ["> " + line for line in cell_source.strip().splitlines()]

        if len(lines) > 10:
            lines = lines[:9] + ["..."]

        print("\n".join(lines), file=sys.stderr)


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


def update_script(path, name, source, *, cell_pattern, cell_marker):
    """Update a cell script"""
    state = UpdateState.wait

    with open(path, "rt") as fobj:
        lines = list(fobj)

    with open(path, "wt") as fobj:
        for line in lines:
            m = cell_pattern.match(line)
            current_name = m.group(1).strip() if m is not None else None

            if state is UpdateState.wait:
                if name == current_name:
                    state = UpdateState.wait_next_cell
                    fobj.write(line)
                    fobj.write(source)

                else:
                    fobj.write(line)

            elif state is UpdateState.wait_next_cell:
                if current_name is not None:
                    fobj.write(line)
                    state = UpdateState.done

            elif state is UpdateState.done:
                fobj.write(line)

        if state is UpdateState.wait:
            fobj.write(f"#{cell_marker} {name}\n")
            fobj.write(source)


class UpdateState(int, enum.Enum):
    wait = enum.auto()
    wait_next_cell = enum.auto()
    done = enum.auto()


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
