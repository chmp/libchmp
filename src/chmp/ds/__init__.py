"""Helpers for data science.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import base64
import bisect
import bz2
import collections
import contextlib
import datetime
import enum
import functools as ft
import gzip
import hashlib
import importlib
import inspect
import io
import itertools as it
import json
import logging
import math
import os.path
import pathlib
import pickle
import re
import sys
import threading
import time

from types import ModuleType
from typing import Any, Callable, Iterable, NamedTuple, Optional, Union

default_sequences = (tuple,)
default_mappings = (dict,)


def reload(*modules_or_module_names: Union[str, ModuleType]) -> Optional[ModuleType]:
    mod = None
    for module_or_module_name in modules_or_module_names:
        if isinstance(module_or_module_name, str):
            module_or_module_name = importlib.import_module(module_or_module_name)

        mod = importlib.reload(module_or_module_name)

    return mod


def reload_from(*paths):
    for path in paths:
        with open(path, "rt") as fobj:
            modules = json.load(fobj)

        for module in modules:
            if not module:
                continue

            reload(module)


def define(func):
    """Execute a function and return its result.

    The idea is to use function scope to prevent pollution of global scope in
    notebooks.

    Usage::

        @define
        def foo():
            return 42

        assert foo == 42

    """
    return func()


def cached(path: str, validate: bool = False):
    """Similar to ``define``, but cache to a file.

    :param path:
        the path of the cache file to use
    :param validate:
        if `True`, always execute the function. The loaded result will be
        passed to the function, when the cache exists. In that case the
        function should return the value to use. If the returned value is
        not identical to the loaded value, the cache is updated with the
        new value.

    Usage::

        @cached('./cache/result')
        def dataset():
            ...
            return result

    or::

        @cached('./cache/result', validate=True)
        def model(result=None):
            if result is not None:
                # running to validate ...

            return result
    """

    def update_cache(result):
        print("save cache", path)
        with open(path, "wb") as fobj:
            pickle.dump(result, fobj)

    def load_cache():
        print("load cache", path)
        with open(path, "rb") as fobj:
            return pickle.load(fobj)

    def decorator(func):
        if os.path.exists(path):
            result = load_cache()

            if not validate:
                return result

            else:
                print("validate")
                new_result = func(result)

                if new_result is not result:
                    update_cache(new_result)

                return new_result

        else:
            print("compute")
            result = func()
            update_cache(result)
            return result

    return decorator


class Object:
    """Dictionary-like namespace object."""

    def __init__(*args, **kwargs):
        self, *args = args

        if len(args) > 1:
            raise ValueError(
                "Object(...) can be called with at " "most one positional argument"
            )

        elif len(args) == 0:
            seed = {}

        else:
            (seed,) = args
            if not isinstance(seed, collections.abc.Mapping):
                seed = vars(seed)

        for k, v in dict(seed, **kwargs).items():
            setattr(self, k, v)

    def __repr__(self):
        return "Object({})".format(
            ", ".join("{}={!r}".format(k, v) for k, v in vars(self).items())
        )

    def __eq__(self, other):
        return type(self) == type(other) and vars(self) == vars(other)

    def __ne__(self, other):
        return not (self == other)


class daterange:
    """A range of dates."""

    start: datetime.date
    end: datetime.date
    step: datetime.timedelta

    @classmethod
    def around(cls, dt, start, end, step=None):
        if not isinstance(start, datetime.timedelta):
            start = datetime.timedelta(days=start)

        if not isinstance(end, datetime.timedelta):
            end = datetime.timedelta(days=end)

        if step is None:
            step = datetime.timedelta(days=1)

        elif not isinstance(step, datetime.timedelta):
            step = datetime.timedelta(days=step)

        return cls(dt + start, dt + end, step)

    def __init__(
        self,
        start: datetime.date,
        end: datetime.date,
        step: Optional[datetime.timedelta] = None,
    ):
        if step is None:
            step = datetime.timedelta(days=1)

        self.start = start
        self.end = end
        self.step = step

    def __len__(self) -> int:
        return len(self._offset_range)

    def __iter__(self) -> Iterable[datetime.date]:
        for offset in self._offset_range:
            yield self.start + datetime.timedelta(days=offset)

    def __contains__(self, item: datetime.date) -> bool:
        return self._offset(item) in self._offset_range

    def __getitem__(self, index: int) -> datetime.date:
        return self.start + datetime.timedelta(days=self._offset_range[index])

    def count(self, item: datetime.date) -> int:
        return 1 if (item in self) else 0

    def index(self, item):
        return self._offset_range.index(self._offset(item))

    def _offset(self, item: datetime.date) -> int:
        return (item - self.start).days

    @property
    def _offset_range(self) -> range:
        return range(0, (self.end - self.start).days, self.step.days)

    def __repr__(self):
        return f"daterange({self.start}, {self.end}, {self.step})"


class undefined_meta(type):
    def __repr__(self):
        return "<undefined>"


class undefined(metaclass=undefined_meta):
    """A sentinel to mark undefined argument values."""

    def __init__(self):
        pass


def first(iterable, default=undefined):
    """Return the first item of an iterable"""
    nth(iterable, 0, default=default)


def last(iterable, default=undefined):
    """Return the last item of an iterable"""
    item = default
    for item in iterable:
        pass

    return item


def nth(iterable, n, default=undefined):
    """Return the nth value in an iterable."""
    for i, item in enumerate(iterable):
        if i == n:
            return item

    return default


def collect(iterable):
    """Collect an iterable of ``key, value`` pairs into a dict of lists."""
    result = {}
    for k, v in iterable:
        result.setdefault(k, []).append(v)

    return result


class cell:
    """No-op context manager to allow indentation of code"""

    def __init__(self, name=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

    def __call__(self, func):
        with self:
            func()


def colorize(items, cmap=None):
    """Given an iterable, yield ``(color, item)`` pairs.

    :param cmap:
        if None the color cycle is used, otherwise it is interpreted as a
        colormap to color the individual items.

        Note: ``items`` is fully instantiated during the iteration. For any
        ``list`` or ``tuple`` item only its first element is used for
        colomapping.

        This procedure allows for example to colormap a pandas Dataframe
        grouped on a number column::

            for c, (_, g) in colorize(df.groupby("g"), cmap="viridis"):
                ...
    """
    if cmap is None:
        cycle = get_color_cycle()
        return zip(it.cycle(cycle), items)

    else:
        items = list(items)

        if not items:
            return iter(())

        keys = [item[0] if isinstance(item, (tuple, list)) else item for item in items]

        return zip(colormap(keys, cmap=cmap), items)


def get_color_cycle(n=None):
    """Return the matplotlib color cycle.

    :param Optional[int] n:
        if given, return a list with exactly n elements formed by repeating
        the color cycle as necessary.

    Usage::

        blue, green, red = get_color_cycle(3)

    """
    import matplotlib as mpl

    cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    if n is None:
        return it.cycle(cycle)

    return list(it.islice(it.cycle(cycle), n))


@contextlib.contextmanager
def mpl_figure(
    n_rows=None,
    n_cols=None,
    n_axes=None,
    dpi=None,
    wspace=1.0,
    hspace=1.5,
    axis_height=2.5,
    axis_width=3.5,
    left_margin=0.5,
    right_margin=0.1,
    top_margin=0.1,
    bottom_margin=0.5,
    title=None,
):
    import matplotlib.pyplot as plt

    n_rows, n_cols, n_axes = _normalize_figure_args(n_rows, n_cols, n_axes)

    width = left_margin + right_margin + (n_cols - 1) * wspace + n_cols * axis_width
    height = bottom_margin + top_margin + (n_rows - 1) * hspace + n_rows * axis_height

    gridspec_kw = dict(
        bottom=bottom_margin / height,
        top=1.0 - top_margin / height,
        left=left_margin / width,
        right=1.0 - right_margin / width,
        wspace=wspace / axis_width,
        hspace=hspace / axis_height,
    )

    _, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width, height),
        gridspec_kw=gridspec_kw,
        dpi=dpi,
    )
    if n_rows == 1 and n_cols == 1:
        axes = (axes,)

    else:
        axes = tuple(axes.flatten())

    used_axes, unused_axes = axes[:n_axes], axes[n_axes:]

    if title is not None:
        plt.suptitle(title)

    for ax in unused_axes:
        ax.remove()

    yield used_axes if n_axes != 1 else used_axes[0]

    # TODO: add support to save the figure, close it, etc..


def _normalize_figure_args(n_rows, n_cols, n_axes):
    has_rows = n_rows is not None
    has_cols = n_cols is not None
    has_axis = n_axes is not None

    if has_rows and has_cols and has_axis:
        pass

    elif has_rows and has_cols and not has_axis:
        n_axes = n_rows * n_cols

    elif has_rows and not has_cols and has_axis:
        n_cols = n_axes // n_rows + ((n_axes % n_rows) != 0)

    elif not has_rows and has_cols and has_axis:
        n_rows = n_axes // n_cols + ((n_axes % n_cols) != 0)

    elif not has_rows and not has_cols and has_axis:
        n_cols = 1
        n_rows = n_axes // n_cols + ((n_axes % n_cols) != 0)

    elif not has_rows and has_cols and not has_axis:
        n_rows = 1
        n_axes = n_rows * n_cols

    elif has_rows and not has_cols and not has_axis:
        n_cols = 1
        n_axes = n_rows * n_cols

    elif not has_rows and not has_cols and not has_axis:
        n_rows = 1
        n_cols = 1
        n_axes = 1

    assert n_axes <= n_rows * n_cols

    return n_rows, n_cols, n_axes


@contextlib.contextmanager
def mpl_axis(
    ax=None,
    *,
    box=None,
    xlabel=None,
    ylabel=None,
    title=None,
    suptitle=None,
    xscale=None,
    yscale=None,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    xformatter: Optional[Callable[[float, float], str]] = None,
    yformatter: Optional[Callable[[float, float], str]] = None,
    legend=None,
    colorbar=None,
    invert: Optional[str] = None,
    grid=None,
    axis=None,
):
    """Set various style related options of MPL.

    :param xformatter:
        if given a formatter for the major x ticks. Should have the
        signature ``(x_value, pos) -> label``.

    :param yformatter:
        See ``xformatter``.

    :param invert:
        if given invert the different axes. Can be `x`, `y`, or `xy`.
    """
    import matplotlib.pyplot as plt

    prev_ax = plt.gca() if plt.get_fignums() else None

    if ax is None:
        _, ax = plt.subplots()

    plt.sca(ax)
    yield ax

    if box is not None:
        plt.box(box)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    if suptitle is not None:
        plt.suptitle(suptitle)

    if xscale is not None:
        plt.xscale(xscale)

    if yscale is not None:
        plt.yscale(yscale)

    # TODO: handle min/max, enlarge ...
    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    if xticks is not None:
        if isinstance(xticks, tuple):
            plt.xticks(*xticks)

        elif isinstance(xticks, dict):
            plt.xticks(**xticks)

        else:
            plt.xticks(xticks)

    if yticks is not None:
        if isinstance(yticks, tuple):
            plt.yticks(*yticks)

        elif isinstance(yticks, dict):
            plt.yticks(**yticks)

        else:
            plt.yticks(yticks)

    if xformatter is not None:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(xformatter))

    if yformatter is not None:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(yformatter))

    if legend is not None and legend is not False:
        if legend is True:
            plt.legend(loc="best")

        elif isinstance(legend, str):
            plt.legend(loc=legend)

        else:
            plt.legend(**legend)

    if colorbar is True:
        plt.colorbar()

    if invert is not None:
        if "x" in invert:
            plt.gca().invert_xaxis()

        if "y" in invert:
            plt.gca().invert_yaxis()

    if grid is not None:
        if not isinstance(grid, list):
            grid = [grid]

        for spec in grid:
            if isinstance(spec, bool):
                _b, _which, _axis = spec, "major", "both"

            elif isinstance(spec, str):
                _b, _which, _axis = True, "major", spec

            elif isinstance(spec, tuple) and len(spec) == 2:
                _b, _which, _axis = True, spec[0], spec[1]

            elif isinstance(spec, tuple):
                _b, _which, _axis = spec

            else:
                raise RuntimeError()

            plt.grid(_b, _which, _axis)

    if axis is not None and axis is not True:
        if axis is False:
            axis = "off"

        plt.axis(axis)

    # restore the previous axis
    if prev_ax is not None:
        plt.sca(prev_ax)


def errorband(data, *, x=None, y, yerr, **kwargs):
    """Plot erros as a band around a line

    Usage::

        df.pipe(errorband, y="mean", yerr="std")

    """
    import matplotlib.pyplot as plt
    import numpy as np

    if x is None:
        xlabel = data.index.name
        x = np.asarray(data.index)

    else:
        xlabel = x
        x = np.asarray(data[x])

    ylabel = y
    y = np.asarray(data[y])
    yerr = np.asarray(data[yerr])

    if yerr.ndim == 2 and yerr.shape[1] == 2:
        yerr_lower, yerr_upper = yerr.T

    else:
        yerr_lower, yerr_upper = yerr, yerr

    (l,) = plt.plot(x, y, **kwargs)

    alpha = l.get_alpha()
    if alpha is None:
        alpha = 1.0

    plt.fill_between(
        x,
        y - yerr_lower,
        y + yerr_upper,
        alpha=0.5 * alpha,
        color=l.get_color(),
        zorder=l.get_zorder() - 1,
    )

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)


def diagonal(df=None, *, x, y, type="scatter", ax=None, **kwargs):
    """Create a diagonal plot

    :param type:
        the type to plot. Possible type "scatter", "hexbin".
    """
    import matplotlib.pyplot as plt

    if df is not None:
        x_data = df[x]
        y_data = df[y]

    else:
        x_data = x
        y_data = y
        x = "x"
        y = "y"

    if ax is None:
        ax = plt.gca()

    vmin = max(x_data.min(), y_data.min())
    vmax = min(x_data.max(), y_data.max())

    plt.sca(ax)

    if type == "scatter":
        plt.plot(x_data, y_data, ".")

    elif type == "hexbin":
        logcount = kwargs.pop("logcount", True)

        kwargs = {
            "gridsize": 31,
            "extent": [vmin, vmax, vmin, vmax],
            **kwargs,
        }

        if logcount:
            kwargs.setdefault("bins", "log")
            kwargs.setdefault("mincnt", 1)

        plt.hexbin(x_data, y_data, **kwargs)
        plt.colorbar()

    plt.plot([vmin, vmax], [vmin, vmax], color="r")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.axis("equal")


def edges(x):
    """Create edges for use with pcolor.

    Usage::

        assert x.size == v.shape[1]
        assert y.size == v.shape[0]
        pcolor(edges(x), edges(y), v)

    """
    import numpy as np

    centers = 0.5 * (x[1:] + x[:-1])
    return np.concatenate(
        ([x[0] - 0.5 * (x[1] - x[0])], centers, [x[-1] + 0.5 * (x[-1] - x[-2])])
    )


def center(u):
    """Compute the center between edges."""
    return 0.5 * (u[1:] + u[:-1])


def axtext(*args, **kwargs):
    """Add a text in axes coordinates (similar ``figtext``).

    Usage::

        axtext(0, 0, 'text')

    """
    import matplotlib.pyplot as plt

    kwargs.update(transform=plt.gca().transAxes)
    plt.text(*args, **kwargs)


def _dict_of_optionals(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


def index_query(obj, expression, scalar=False):
    """Execute a query expression on the index and return matching rows.

    :param scalar:
        if True, return only the first item. Setting ``scalar=True``
        raises an error if the resulting object has have more than one
        entry.
    """
    res = obj.loc[obj.index.to_frame().query(expression).index]

    if scalar:
        assert res.size == 1
        return res.iloc[0]

    return res


def select(_df, *args, **kwargs):
    import pandas as pd

    parts = [
        _df[[*args]],
        _df.assign(**kwargs)[[*kwargs]],
    ]
    return pd.concat(parts, axis=1)


def query(_df, *args, **kwargs):
    """Filter a dataframe.

    Usage::

        df.pipe(query, lambda df: df["col"] == "foo")
        df.pipe(query, col="foo", bar="baz")

    """
    import numpy as np

    sel = True
    for arg in args:
        sel = np.logical_and(sel, arg(_df))

    for column, value in kwargs.items():
        if isinstance(value, tuple):
            sel = np.logical_and(sel, _df[column].isin(value))

        else:
            sel = np.logical_and(sel, _df[column] == value)

    return _df[sel]


def fix_categories(
    s, categories=None, other_category=None, inplace=False, groups=None, ordered=False
):
    """Fix the categories of a categorical series.

    :param pd.Series s:
        the series to normalize

    :param Optional[Iterable[Any]] categories:
        the categories to keep. The result will have categories in the
        iteration order of this parameter. If not given but ``groups`` is
        passed, the keys of ``groups`` will be used, otherwise the existing
        categories of ``s`` will be used.

    :param Optional[Any] other_category:
        all categories to be removed wil be mapped to this value, unless they
        are specified specified by the ``groups`` parameter. If given and not
        included in the categories, it is appended to the given categories.
        For a custom order, ensure it is included in ``categories``.

    :param bool inplace:
        if True the series will be modified in place.

    :param Optional[Mapping[Any,Iterable[Any]]] groups:
        if given, specifies which categories to replace by which in the form
        of ``{replacement: list_of_categories_to_replace}``.

    :param bool ordered:
        if True the resulting series will have ordered categories.
    """
    import pandas.api.types as pd_types

    if not inplace:
        s = s.copy()

    if not pd_types.is_categorical(s):
        if inplace:
            raise ValueError("cannot change the type inplace")

        s = s.astype("category")

    if categories is None:
        if groups is not None:
            categories = list(groups.keys())

        else:
            categories = list(s.cat.categories)

    categories = list(categories)
    inital_categories = set(s.cat.categories)

    if other_category is not None and other_category not in categories:
        categories = categories + [other_category]

    additions = [c for c in categories if c not in inital_categories]
    removals = [c for c in inital_categories if c not in categories]

    if groups is None:
        groups = {}

    else:
        groups = {k: set(v) for k, v in groups.items()}

    remapped = {c for group in groups.values() for c in group}

    dangling_categories = {*removals} - {*remapped}
    if dangling_categories:
        if other_category is None:
            raise ValueError(
                "dangling categories %s found, need other category to assign"
                % dangling_categories
            )

        groups.setdefault(other_category, set()).update(set(removals) - set(remapped))

    if additions:
        s.cat.add_categories(additions, inplace=True)

    for replacement, group in groups.items():
        s[s.isin(group)] = replacement

    if removals:
        s.cat.remove_categories(removals, inplace=True)

    s.cat.reorder_categories(categories, inplace=True, ordered=ordered)

    return s


def find_high_frequency_categories(s, min_frequency=0.02, n_max=None):
    """Find categories with high frequency.

    :param float min_frequency:
        the minimum frequency to keep

    :param Optional[int] n_max:
        if given keep at most ``n_max`` categories. If more are present after
        filtering for minimum frequency, keep the highest ``n_max`` frequency
        columns.
    """
    assert 0.0 < min_frequency < 1.0
    s = s.value_counts(normalize=True).pipe(lambda s: s[s > min_frequency])

    if n_max is None:
        return list(s.index)

    if len(s) <= n_max:
        return s

    return list(s.sort_values(ascending=False).iloc[:n_max].index)


def as_frame(*args, **kwargs):
    """Build a dataframe from kwargs or positional args.

    Note, functions can be passed as kwargs. They will be evaluated with the
    current dataframe and their result assigned to the named column. For
    example::

        as_frame(
            x=np.random.uniform(-3, 3, 1_000),
            y=lambda df: np.random.normal(df["x"], 0.5),
        )

    """
    import pandas as pd

    if args and kwargs:
        raise ValueError(
            "as_frame cannot be called with args or kwargs at the same time"
        )

    args = list(args if args else ())
    return pd.DataFrame(args).assign(**kwargs)


def setdefaultattr(obj, name, value):
    """``dict.setdefault`` for attributes"""
    if not hasattr(obj, name):
        setattr(obj, name, value)

    return getattr(obj, name)


def transform_args(func, args, kwargs, transform, **transform_args):
    """Transform the arguments of the function.

    The arguments are normalized into a dictionary before being passed to the
    transform function. The return value is a tuple of ``args, kwargs`` ready to
    be passed to ``func``.
    """
    bound_args = inspect.signature(func).bind(*args, **kwargs)

    transformed_args = transform(dict(bound_args.arguments), **transform_args)
    bound_args.arguments.update(**transformed_args)

    return bound_args.args, bound_args.kwargs


def szip(
    iterable_of_objects,
    sequences=default_sequences,
    mappings=default_mappings,
    combine=list,
):
    """Zip but for deeply nested objects.
    For a list of nested set of objects return a nested set of list.
    """
    iterable_of_objects = iter(iterable_of_objects)

    try:
        first = next(iterable_of_objects)

    except StopIteration:
        return None

    # build a scaffold into which the results are appended
    # NOTE: the target lists must not be confused with the structure, use a
    # schema object as an unambiguous marker.
    schema = smap(lambda _: None, first, sequences=sequences, mappings=mappings)
    target = smap(lambda _: [], schema, sequences=sequences, mappings=mappings)

    for obj in it.chain([first], iterable_of_objects):
        smap(
            lambda _, t, o: t.append(o),
            schema,
            target,
            obj,
            sequences=sequences,
            mappings=mappings,
        )

    return smap(
        lambda _, o: combine(o),
        schema,
        target,
        sequences=sequences,
        mappings=mappings,
    )


def flatten_with_index(obj, sequences=default_sequences, mappings=default_mappings):
    counter = iter(it.count())
    flat = []

    def _build(item):
        flat.append(item)
        return next(counter)

    index = smap(_build, obj, sequences=sequences, mappings=mappings)
    return index, flat


def unflatten(index, obj, sequences=default_sequences, mappings=default_mappings):
    obj = list(obj)
    return smap(lambda idx: obj[idx], index, sequences=sequences, mappings=mappings)


def smap(func, arg, *args, sequences=default_sequences, mappings=default_mappings):
    """A structured version of map.
    The structure is taken from the first arguments.
    """
    return _smap(func, arg, *args, path="$", sequences=sequences, mappings=mappings)


def _smap(
    func, arg, *args, path, sequences=default_sequences, mappings=default_mappings
):
    try:
        if isinstance(arg, sequences):
            return type(arg)(
                _smap(
                    func,
                    *co,
                    path=f"{path}.{idx}",
                    sequences=sequences,
                    mappings=mappings,
                )
                for idx, *co in zip(it.count(), arg, *args)
            )

        elif isinstance(arg, mappings):
            return type(arg)(
                (
                    k,
                    _smap(
                        func,
                        arg[k],
                        *(obj[k] for obj in args),
                        path=f"{path}.k",
                        sequences=sequences,
                        mappings=mappings,
                    ),
                )
                for k in arg
            )

        else:
            return func(arg, *args)

    # pass through any exceptions in smap without further annotations
    except SApplyError:
        raise

    except Exception as e:
        raise SApplyError(f"Error in sappend at {path}: {e}") from e


def copy_structure(
    template, obj, sequences=default_sequences, mappings=default_mappings
):
    """Arrange ``obj`` into the structure of ``template``.

    :param template:
        the object of which to copy the structure
    :param obj:
        the object which to arrange into the structure. If it is
        already structured, the template structure and its structure
        must be the same or a value error is raised
    """

    template_schema = smap(
        lambda _: None, template, sequences=sequences, mappings=mappings
    )
    obj_schema = smap(lambda _: None, obj, sequences=sequences, mappings=mappings)

    if obj_schema is not None:
        if obj_schema != template_schema:
            raise ValueError("Misaligned structures")

        return obj

    return smap(lambda _: obj, template_schema, sequences=sequences, mappings=mappings)


def assert_has_schema(
    nested_obj, expected_schema, sequences=default_sequences, mappings=default_mappings
):
    actual_schema = smap(
        lambda _: None, nested_obj, sequences=sequences, mappings=mappings
    )

    if actual_schema != expected_schema:
        raise AssertionError(
            f"Schemas do not match: {actual_schema} != {expected_schema}"
        )


class SApplyError(Exception):
    pass


def json_numpy_default(obj):
    """A default implementation for ``json.dump`` that deals with numpy datatypes."""
    import numpy as np

    int_types = tuple(
        getattr(np, dt)
        for dt in (
            "int0",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint0",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        )
        if hasattr(np, dt)
    )

    float_types = tuple(
        getattr(np, dt)
        for dt in ("float16", "float32", "float64", "float128")
        if hasattr(np, dt)
    )

    if isinstance(obj, int_types):
        return int(obj)

    elif isinstance(obj, float_types):
        return float(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Cannot convert type of {type(obj).__name__}")


def piecewise_linear(x, y, xi):
    return _piecewise(_linear_interpolator, x, y, xi)


def piecewise_logarithmic(x, y, xi=None):
    return _piecewise(_logarithmic_interpolator, x, y, xi)


def _linear_interpolator(u, y0, y1):
    return y0 + u * (y1 - y0)


def _logarithmic_interpolator(u, y0, y1):
    return (y0 ** (1 - u)) * (y1 ** u)


def _piecewise(interpolator, x, y, xi):
    assert len(x) == len(y)
    interval = bisect.bisect_right(x, xi)

    if interval == 0:
        return y[0]

    if interval >= len(x):
        return y[-1]

    u = (xi - x[interval - 1]) / (x[interval] - x[interval - 1])
    return interpolator(u, y[interval - 1], y[interval])


def pd_has_ordered_assign():
    import pandas as pd

    py_major, py_minor, *_ = sys.version_info
    pd_major, pd_minor, *_ = pd.__version__.split(".")
    pd_major = int(pd_major)
    pd_minor = int(pd_minor)

    return (py_major, py_minor) >= (3, 6) and (pd_major, pd_minor) >= (0, 23)


@contextlib.contextmanager
def patch(obj, **assignments):
    """Monkey patch an object.

    Usage::

        import sys
        with patch(sys, argv=["dummy.py", "foo", "bar]):
            args = parser.parse_args()

    After the block, the original values will be restored. Note: If an
    attribute was previously not defined, it will be deleted after the block.
    """
    prev_values = [
        (attr, hasattr(obj, attr), getattr(obj, attr, None)) for attr in assignments
    ]

    try:
        for attr, value in assignments.items():
            setattr(obj, attr, value)

        yield

    finally:
        for attr, had_value, prev_value in prev_values:
            if had_value:
                setattr(obj, attr, prev_value)

            else:
                delattr(obj, attr)


def groupby(a):
    """Group a numpy array.

    Usage::

        for idx in grouped(arr):
            ...

    """
    starts, ends, indices = _groupby(a)
    return Grouped(
        starts=starts,
        ends=ends,
        indices=indices,
    )


class Grouped:
    def __init__(self, starts, ends, indices):
        self.starts = starts
        self.ends = ends
        self.indices = indices

    def __iter__(self):
        for i in range(len(self.starts)):
            yield self.indices[self.starts[i] : self.ends[i]]


def _groupby(a):
    import numpy as np

    offset = 0
    starts = []
    ends = []
    indices = np.empty(len(a), dtype="uint64")

    for v in np.unique(a):
        (idx,) = np.nonzero(a == v)

        indices[offset : offset + len(idx)] = idx
        starts += [offset]
        ends += [offset + len(idx)]

        offset += len(idx)

    starts = np.asarray(starts, dtype="uint64")
    ends = np.asarray(ends, dtype="uint64")

    return starts, ends, indices


def timed(tag=None, level=logging.INFO):
    """Time a codeblock and log the result.

    Usage::

        with timed():
            long_running_operation()

    The returned result can be used to estimate the remaining runtime::

        with timed() as timer:
            timer(0.5)

    :param any tag:
        an object used to identify the timed code block. It is printed with
        the time taken.
    """
    return _TimedContext(
        message=("[TIMING] %s s" if tag is None else "[TIMING] {} %s s".format(tag)),
        logger=_get_caller_logger(),
        level=level,
    )


# use a custom contextmanager to control stack level for _get_caller_logger
class _TimedContext:
    def __init__(self, logger, message, level):
        self.logger = logger
        self.message = message
        self.level = level
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.logger.log(self.level, self.message, end - self.start)

    def __call__(self, u):
        u = min(max(u, 0.001), 1)
        res = (time.time() - self.start) * (1 - u) / u
        return PrettySeconds(res)


class PrettySeconds:
    def __init__(self, seconds):
        self.seconds = seconds

    def __float__(self):
        return self.seconds

    def __str__(self):
        return format(self)

    def __format__(self, format_spec):
        sign = "" if self.seconds > -1e-5 else "-"
        time_delta = abs(self.seconds)

        d = dict(
            weeks=int(time_delta // (7 * 24 * 60 * 60)),
            days=int(time_delta % (7 * 24 * 60 * 60) // (24 * 60 * 60)),
            hours=int(time_delta % (24 * 60 * 60) // (60 * 60)),
            minutes=int(time_delta % (60 * 60) // 60),
            seconds=time_delta % 60,
            sign=sign,
        )

        if d["weeks"] > 0:
            return "{sign}{weeks}w {days}d".format(**d)

        elif d["days"] > 0:
            return "{sign}{days}d {hours}h".format(**d)

        elif d["hours"] > 0:
            return "{sign}{hours}h {minutes}m".format(**d)

        elif d["minutes"] > 0:
            return "{sign}{minutes}m {seconds:.0f}s".format(**d)

        else:
            return "{sign}{seconds:.1f}s".format(**d)


class Debouncer:
    def __init__(self, interval, *, now=time.time):
        self.last_invocation = 0
        self.interval = interval
        self.now = now

    def should_run(self, now=None):
        if self.interval is False:
            return True

        if now is None:
            now = self.now()

        return now > self.last_invocation + self.interval

    def invoked(self, now=None):
        if now is None:
            now = self.now()

        self.last_invocation = now


def _get_caller_logger(depth=2):
    stack = inspect.stack()

    if depth >= len(stack):  # pragma: no cover
        return logging.getLogger(__name__)

    # NOTE: python2 returns raw tuples, index 0 is the frame
    frame = stack[depth][0]
    name = frame.f_globals.get("__name__")
    return logging.getLogger(name)


def _setup_status(func):
    def config(width=None, interval=None):
        if width is not None:
            func.width = width

        if interval is not None:
            func.debouncer.interval = interval

    func.debouncer = Debouncer(interval=0.5)
    func.width = 120
    func.config = config

    return func


@_setup_status
def status(**items):
    """Print (and update) a status line

    Usage::

        status(epoch=epoch, loss=("{:.2f}", loss), complex=callable)

    The print call is debounced to only execute only twice per second and to
    take up at most 120 characters. To change these settings use the `config`
    function::

        status.config(width=80, interval=2)

    """
    if not status.debouncer.should_run():
        return

    status.debouncer.invoked()

    kv_pairs = []

    for key, val in items.items():
        if callable(val):
            val = val()

        if isinstance(val, tuple):
            fmt, val = val

        else:
            fmt, val = "{}", val

        kv_pairs.append(("{}=" + fmt).format(key, val))

    status_line = " ".join(kv_pairs)
    print(status_line[: status.width].ljust(status.width), end="\r")


def find_categorical_columns(df):
    """Find all categorical columns in the given dataframe."""
    import pandas.api.types as pd_types

    return [k for k, dtype in df.dtypes.items() if pd_types.is_categorical_dtype(dtype)]


def colormap(x, cmap="coolwarm", center=True, vmin=None, vmax=None, norm=None):
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    x = np.asarray(x)

    if norm is None:
        norm = colors.NoNorm()

    if vmin is None:
        vmin = np.min(x)

    if vmax is None:
        vmax = np.max(x)

    if center:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax

    x = norm(x)
    x = np.clip((x - vmin) / (vmax - vmin), 0, 1)

    return cm.get_cmap(cmap)(x)


def expand(low, high, change=0.05):
    center = 0.5 * (low + high)
    delta = 0.5 * (high - low)
    return (center - (1 + 0.5 * change) * delta, center + (1 + 0.5 * change) * delta)


# ########################################################################## #
#                             I/O Methods                                    #
# ########################################################################## #


def magic_open(p, mode, *, compression=None, atomic=False):
    """Open compressed and  uncompressed files with a unified interface.

    :param compression:
        the compression desired. If not given it will be autodetected from the
        filename.
    """
    # return file-like objects unchanged
    if not isinstance(p, (pathlib.Path, str)):
        return p

    assert atomic is False, "Atomic operations not yet supported"
    opener = _get_opener(p, compression)
    return opener(p, mode)


def _get_opener(p, compression):
    if compression is None:
        sp = str(p)

        if sp.endswith(".bz2"):
            compression = "bz2"

        elif sp.endswith(".gz"):
            compression = "gz"

        else:
            compression = "none"

    openers = {"bz2": bz2.open, "gz": gzip.open, "gzip": gzip.open, "none": open}
    return openers[compression]


# ########################################################################## #
#                               TQDM Helpers                                 #
# ########################################################################## #


def clear_tqdm():
    """Close any open TQDM instances to prevent display errors"""
    import tqdm

    # NOTE: the _instances attribute is only set on first use of tqdm
    for inst in list(getattr(tqdm.tqdm, "_instances", [])):
        inst.close()


# ###################################################################### #
# #                                                                    # #
# #                 Deterministic Random Number Generation             # #
# #                                                                    # #
# ###################################################################### #

maximum_15_digit_hex = float(0xFFF_FFFF_FFFF_FFFF)
max_32_bit_integer = 0xFFFF_FFFF


def sha1(obj):
    """Create a hash for a json-encode-able object"""
    return int(str_sha1(obj)[:15], 16)


def str_sha1(obj):
    s = json.dumps(obj, indent=None, sort_keys=True, separators=(",", ":"))
    s = s.encode("utf8")
    return hashlib.sha1(s).hexdigest()


def random(obj):
    """Return a random float in the range [0, 1)"""
    return min(sha1(obj) / maximum_15_digit_hex, 0.999_999_999_999_999_9)


def uniform(obj, a, b):
    return a + (b - a) * random(obj)


def randrange(obj, *range_args):
    r = range(*range_args)
    # works up to a len of 9007199254749999, rounds down afterwards
    i = int(random(obj) * len(r))
    return r[i]


def randint(obj, a, b):
    return randrange(obj, a, b + 1)


def np_seed(obj):
    """Return a seed usable by numpy."""
    return [randrange((obj, i), max_32_bit_integer) for i in range(10)]


def tf_seed(obj):
    """Return a seed usable by tensorflow."""
    return randrange(obj, max_32_bit_integer)


def std_seed(obj):
    """Return a seed usable by python random module."""
    return str_sha1(obj)


def shuffled(obj, l):
    l = list(l)
    shuffle(obj, l)
    return l


def shuffle(obj, l):
    """Shuffle ``l`` in place using Fisher–Yates algorithm.

    See: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    """
    n = len(l)
    for i in range(n - 1):
        j = randrange((obj, i), i, n)
        l[i], l[j] = l[j], l[i]


# ########################################################################### #
#                                                                             #
#                     Helper for datetime handling in pandas                  #
#                                                                             #
# ########################################################################### #
def timeshift_index(obj, dt):
    """Return a shallow copy of ``obj`` with its datetime index shifted by ``dt``."""
    obj = obj.copy(deep=False)
    obj.index = obj.index + dt
    return obj


def to_start_of_day(s):
    """Return the start of the day for the datetime given in ``s``."""
    import pandas as pd

    s = pd.to_datetime({"year": s.dt.year, "month": s.dt.month, "day": s.dt.day})
    s = pd.Series(s)
    return s


def to_time_in_day(s, unit=None):
    """Return the timediff relative to the start of the day of ``s``."""
    import pandas as pd

    s = s - to_start_of_day(s)
    return s if unit is None else s / pd.to_timedelta(1, unit=unit)


def to_start_of_week(s):
    """Return the start of the week for the datetime given ``s``."""
    s = to_start_of_day(s)
    return s - s.dt.dayofweek * datetime.timedelta(days=1)


def to_time_in_week(s, unit=None):
    """Return the timedelta relative to weekstart for the datetime given in ``s``."""
    import pandas as pd

    s = s - to_start_of_week(s)
    return s if unit is None else s / pd.to_timedelta(1, unit=unit)


def to_start_of_year(s):
    """Return the start of the year for the datetime given in ``s``."""
    import pandas as pd

    s = pd.to_datetime({"year": s.dt.year, "month": 1, "day": 1})
    s = pd.Series(s)
    return s


def to_time_in_year(s, unit=None):
    """Return the timediff relative to the start of the year for ``s``."""
    import pandas as pd

    s = s - to_start_of_year(s)
    return s if unit is None else s / pd.to_timedelta(1, unit=unit)
