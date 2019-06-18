import weakref
import ast
import functools as ft
import uuid

from ipywidgets import DOMWidget
from traitlets import Unicode


def _jupyter_nbextension_paths():
    """Return metadata for the jupyter-vega nbextension."""
    return [
        dict(
            section="notebook",
            src=".",
            dest="chmp-widgets",
            require="chmp-widgets/index",
        )
    ]


class FocusCell(DOMWidget):
    """A widget to hide all other cells, but the one containing this widget.

    Usage::

        # in a notebook cell
        widget = FocusCell()
        widget
    """

    _view_name = Unicode("FocusCell").tag(sync=True)
    _view_module = Unicode("nbextensions/chmp-widgets/widgets").tag(sync=True)
    _view_module_version = Unicode("0.1.0").tag(sync=True)


class WidgetRegistry:
    """Register an retrieve widgets by name.

    Usage::

        registry = WidgetRegistry()

        widget = HBox([
            registry("label", Label()),
        ])

        registry.label.value = "Hello World"

    """

    def __init__(self):
        self.widgets = {}

    def __call__(self, key, widget):
        self.widgets[key] = widget
        return widget

    def __getattr__(self, key):
        try:
            return self.widgets[key]

        except KeyError as _:
            raise AttributeError(key)

    def __dir__(self):
        return [*super().__dir__(), *self.widgets]


class PersistentDatasets:
    """Helper to keep data between the front and backend consistent with VegaWidget.

    Usage::

        # construct the datasets object
        datasets = PersistentDatasets()

        # notify the widget of any changes in the timeseries dataset
        datasets.bind(widget, ["timeseries"])

        # update the dataset
        datasets.update(
            "timeseries",
            # insert new data points
            insert=[
                {"date": "2019-05-03", "value": 2.0},
                {"date": "2019-05-04", "value": 4.0},
            ],
            # specify a JS expression, it will be rewritten into python and
            # also run in the kernel.
            remove="datum.date <  '2019-04-01'",
        )

        # get the current python state of the dataset
        datasets.get("timeseries")

        # clear a single (or all datasets)
        datasets.clear("timeseries")
        datasets.clear_all()

    """

    def __init__(self, datasets=None):
        if datasets is None:
            datasets = {}

        self.datasets = dict(datasets)
        self.widgets = {}

    def bind(self, widget, datasets):
        for key in datasets:
            ref = weakref.ref(widget, self._update_widgets)
            self.widgets.setdefault(key, []).append(ref)

    def _update_widgets(self, _):
        for key in datasets:
            self.widgets[key] = [ref for ref in self.widgets[key] if ref() is not None]

        self.widgets = {key: refs for key, refs in self.widgets.items() if refs}

    def clear(self, key):
        if key in self.datasets:
            self.datasets[key] = []

        for widget in self.widgets.get(key, []):
            widget = widget()
            if widget is not None:
                widget.update(key, remove="true")

    def clear_all(self):
        for key in {*self.datasets, *self.widgets}:
            self.clear(key)

    def get(self, key):
        return {"name": key, "values": self.datasets.get(key, [])}

    def update(self, key, *, remove=None, insert=None):
        for widget in self.widgets.get(key, []):
            widget = widget()
            if widget is not None:
                widget.update(key, remove=remove, insert=insert)

        self._update_self(key, remove=remove, insert=insert)

    def _update_self(self, key, remove=None, insert=None):
        if insert is None and remove is None:
            return

        data = self.datasets.get(key, [])

        if remove is not None:
            remove_expr = JSExpr(["datum"], remove)
            data = [item for item in data if not remove_expr(JSObj(item))]

        if insert is not None:
            data = data + list(insert)

        self.datasets[key] = data


class JSExpr:
    """Wrapper around javascript exression as a python callable."""

    def __init__(self, argnames, expr):
        self.argnames = argnames
        self.expr = expr
        self.code = self.build_code(expr)

    @staticmethod
    def build_code(expr, filename=None):
        try:
            import pyjsparser

        except ImportError:
            raise RuntimeError("JSExpr requires pyjsparser to be installed.")

        if filename is None:
            filename = "<js_to_py:{}>".format(uuid.uuid4())

        js_ast = pyjsparser.parse(expr)
        py_ast = _transform_js_to_python(js_ast)
        return compile(py_ast, filename, mode="eval")

    def __call__(self, *args):
        scope = dict(zip(self.argnames, args))
        return eval(self.code, scope)

    def __repr__(self):
        return "JSExpr({!r}, {!r})".format(self.argnames, self.expr)


class JSObj:
    """A wrapper around a dictionary mimicking javascripts getattr behavior."""

    def __init__(*args, **kwargs):
        if len(args) == 2:
            self, obj = args

        else:
            self, = args
            obj = {}

        assert isinstance(obj, dict)
        self.obj = dict(obj, **kwargs)

    def __getattr__(self, key):
        return self.obj[key]

    def __getitem__(self, key):
        return self.obj[key]


def _dispatch_on_type(func):
    """A helper for building the js to py translator"""
    registry = {}

    @ft.wraps(func)
    def wrapper(obj):
        handler = registry.get(obj["type"], func)
        return handler(obj)

    def register(key):
        def decorator(func):
            registry[key] = func

        return decorator

    wrapper.register = register

    return wrapper


@_dispatch_on_type
def _transform_js_to_python(obj):
    """Convert the AST generated by pyjsparse into a pytohn AST."""
    raise ValueError("Cannot handle {}".format(obj["type"]))


@_transform_js_to_python.register("Program")
def _transform_js_to_python_program(obj):
    is_single_expression = (len(obj["body"]) == 1) and (
        obj["body"][0]["type"] == "ExpressionStatement"
    )

    if not is_single_expression:
        raise ValueError("Can only translate single expression statements")

    expr_ast = _transform_js_to_python(obj["body"][0]["expression"])
    return ast.Expression(body=expr_ast, lineno=0, col_offset=0)


@_transform_js_to_python.register("MemberExpression")
def _transform_js_to_python_member_expression(obj):
    value = _transform_js_to_python(obj["object"])

    if obj["property"]["type"] == "Identifier":
        attr = obj["property"]["name"]
        return ast.Attribute(
            value=value, attr=attr, ctx=ast.Load(), lineno=0, col_offset=0
        )

    else:
        key = _transform_js_to_python(obj["property"])

        return ast.Subscript(
            value=value,
            slice=ast.Index(value=key, lineno=0, col_offset=0),
            ctx=ast.Load(),
            lineno=0,
            col_offset=0,
        )


@_transform_js_to_python.register("BinaryExpression")
def _transform_js_to_python_binary_epxression(obj):
    left = _transform_js_to_python(obj["left"])
    right = _transform_js_to_python(obj["right"])
    op = obj["operator"]

    compapare_ops = {
        "<": ast.Lt,
        ">": ast.Gt,
        "<=": ast.LtE,
        ">=": ast.GtE,
        "==": ast.Eq,
        "!=": ast.NotEq,
    }

    if op in compapare_ops:
        return ast.Compare(
            left=left,
            ctx=ast.Load(),
            ops=[compapare_ops[op]()],
            comparators=[right],
            lineno=0,
            col_offset=0,
        )

    else:
        raise NotImplementedError(obj["operator"])


@_transform_js_to_python.register("Identifier")
def _transform_js_to_python_identifier(obj):
    return ast.Name(id=obj["name"], ctx=ast.Load(), lineno=0, col_offset=0)


@_transform_js_to_python.register("Literal")
def _transform_js_to_python_literal(obj):
    if isinstance(obj["value"], str):
        return ast.Str(s=obj["value"], lineno=0, col_offset=0)

    elif isinstance(obj["value"], float):
        try:
            n = int(obj["raw"])

        except ValueError:
            n = float(obj["raw"])

        return ast.Num(n=n, lineno=0, col_offset=0)

    elif isinstance(obj["value"], bool):
        return ast.NameConstant(value=obj["value"], lineno=0, col_offset=0)

    elif obj["value"] is None:
        return ast.NameConstant(value=None, lineno=0, col_offset=0)

    else:
        raise ValueError(
            "Cannot handle literal of type {}".format(type(obj["value"]).__name__)
        )
