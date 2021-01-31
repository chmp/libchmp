from IPython import get_ipython
from IPython.core.magic import (
    Magics,
    magics_class,
    line_magic,
    cell_magic,
    line_cell_magic,
)

from . import update_script, CellScript


@magics_class
class CscMagics(Magics):
    @classmethod
    def register(cls, shell=None):
        if shell is None:
            shell = get_ipython()

        shell.register_magics(cls)

    @line_cell_magic("csc[load]")
    def load(self, line, cell=None):
        "Magic that works both as %lcmagic and as %%lcmagic"
        script = _get_current_script()
        cell_name = line.strip()
        source = script.get(cell_name)

        self._update_nb_cell(line, source)

    @cell_magic("csc[run]")
    def run(self, line, cell):
        # TODO: check whether the second line has save call, if not insert it
        script = _get_current_script()
        script.exec(cell)

    @cell_magic("csc[save]")
    def save(self, line, cell):
        script = _get_current_script()
        cell_name = line.strip()

        update_script(
            script.path,
            cell_name,
            cell,
            cell_pattern=script.cell_pattern,
            cell_marker=script.cell_marker,
        )
        self._update_nb_cell(line, cell.splitlines())

    def _update_nb_cell(self, line, source):
        source = [
            f"%%csc[run]",
            f"#%%csc[save] {line}",
            *source,
        ]
        self.shell.set_next_input("\n".join(source), replace=True)


def _get_current_script(scope=None):
    scope = _ensure_valid_scope(scope)

    name = _get_current_script_name(scope)
    return vars(scope)[name]


def _get_current_script_name(scope=None):
    scope = _ensure_valid_scope(scope)

    cands = [name for name, obj in vars(scope).items() if isinstance(obj, CellScript)]

    if not cands:
        raise RuntimeError("Did not find a cell script in the global scope")

    elif len(cands) > 1:
        raise RuntimeError("Found mutliple candidates {}", cands)

    return cands[0]


def _ensure_valid_scope(scope=None):
    if scope is None:
        import __main__ as scope

    return scope
