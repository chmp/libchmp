import argparse
import inspect
import pathlib

import pytest

from chmp.csc import CellScript


def test_example(tmpdir):
    path = pathlib.Path(tmpdir) / "example.py"
    path.write_text(example_script.format(marker="%"))

    example = CellScript(path, cell_marker="%", verbose=False)
    assert example.list() == [None, "foo", "bar", "baz"]

    example.run("foo")
    assert example.ns.foo == 13

    with pytest.raises(NameError):
        example.run("baz")

    example.run("bar")
    assert example.ns.bar == 21

    example.run("baz")
    assert example.ns.baz == 42

    # inject values at start
    example.run("baz", bar=23)
    assert example.ns.bar == 23
    assert example.ns.baz == 46


def test_different_marker(tmpdir):
    path = pathlib.Path(tmpdir) / "example.py"
    path.write_text(example_script.format(marker="%%"))

    example = CellScript(path, cell_marker="%%")
    assert example.list() == [None, "foo", "bar", "baz"]


def test_exports_to_dict(tmpdir):
    path = pathlib.Path(tmpdir) / "example.py"
    path.write_text(example_script.format(marker="%"))

    exported = argparse.Namespace()

    example = CellScript(path, cell_marker="%", verbose=False, export_ns=exported)
    example.export(exported_foo="foo")

    example.run("foo")
    assert example.ns.foo == 13

    expected_exported = argparse.Namespace(exported_foo=13)
    assert exported == expected_exported


example_script = """# unmarked cell

#{marker} foo
foo = 13

#{marker} bar
bar = 21

#{marker} baz
baz = 2 * bar"""


def test_getsource(tmpdir):
    func = "def foo():\n    return 42\n"

    path = pathlib.Path(tmpdir) / "example.py"
    path.write_text(getsource_script.format(func=func))

    example = CellScript(path, cell_marker="%", verbose=False)
    example.run("Cell1")

    actual = inspect.getsource(example.ns.foo)
    assert actual == func


getsource_script = """

#% Cell1
a = 13

{func}

#% Cell2
b = a + 2
"""
