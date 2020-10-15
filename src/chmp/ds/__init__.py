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

try:
    from sklearn.base import (
        BaseEstimator,
        TransformerMixin,
        ClassifierMixin,
        RegressorMixin,
    )

except ImportError:
    from ._import_compat import (  # typing: ignore
        BaseEstimator,
        TransformerMixin,
        ClassifierMixin,
        RegressorMixin,
    )

    _HAS_SK_LEARN = False


else:
    _HAS_SK_LEARN = True


try:
    from daft import PGM

except ImportError:
    from ._import_compat import PGM  # typing: ignore

    _HAS_DAFT = False

else:
    _HAS_DAFT = True


def reload(*modules_or_module_names: Union[str, ModuleType]) -> Optional[ModuleType]:
    mod = None
    for module_or_module_name in modules_or_module_names:
        if isinstance(module_or_module_name, str):
            module_or_module_name = importlib.import_module(module_or_module_name)

        mod = importlib.reload(module_or_module_name)

    return mod


def import_object(obj):
    def _import_obj(obj):
        module, _, name = obj.partition(":")
        module = importlib.import_module(module)
        return getattr(module, name)

    return sapply(_import_obj, obj)


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
            if not isinstance(seed, collections.Mapping):
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
    """Sentinel class"""

    pass


def first(iterable, default=undefined):
    """Return the first item of an iterable"""
    for item in iterable:
        return item

    return default


def last(iterable, default=undefined):
    """Return the last item of an iterable"""
    item = default
    for item in iterable:
        pass

    return item


def nth(iterable, n, default=undefined):
    for i, item in enumerate(iterable):
        if i == n:
            return item

    return default


def item(iterable, default=undefined):
    """Given a single item iterable return this item."""
    found = undefined

    for item in iterable:
        if found is not undefined:
            raise ValueError("More than one value to unpack")

        found = item

    if found is not undefined:
        return found

    if default is not undefined:
        return default

    raise ValueError("Need at least one item or a default")


def collect(iterable):
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
    n_axis=None,
    wspace=1.0, 
    hspace=1.5,
    axis_height=2.5,
    axis_width=3.5,
    left_margin=0.5, 
    right_margin=0.1, 
    top_margin=0.1, 
    bottom_margin=0.5,
    title=None
):
    import matplotlib.pyplot as plt

    n_rows, n_cols, n_axis = _normalize_figure_args(n_rows, n_cols, n_axis)
    
    width = left_margin + right_margin + (n_cols - 1) * wspace + n_cols * axis_width
    height = bottom_margin + top_margin + (n_rows - 1) * hspace + n_rows * axis_height
    
    gridspec_kw=dict(
        bottom=bottom_margin / height,
        top=1.0 - top_margin / height,
        left=left_margin / width,
        right=1.0 - right_margin / width,
        wspace=wspace / axis_width,
        hspace=hspace / axis_height,
    )
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), gridspec_kw=gridspec_kw)
    
    if title is not None:
        plt.suptitle(title)
    
    yield axes.flatten()[:n_axis]


def _normalize_figure_args(n_rows, n_cols, n_axis):
    has_rows = n_rows is not None
    has_cols = n_cols is not None
    has_axis = n_axis is not None
    
    if has_rows and has_cols and has_axis:
        pass
    
    elif has_rows and has_cols and not has_axis:
        n_axis = n_rows * n_cols
        
    elif has_rows and not has_cols and has_axis:
        n_cols = n_axis // n_rows + ((n_axis % n_rows) != 0)
    
    elif not has_rows and has_cols and has_axis:
        n_rows = n_axis // n_cols + ((n_axis % n_cols) != 0)
    
    elif not has_rows and not has_cols and has_axis:
        n_cols = 1
        n_rows = n_axis // n_cols + ((n_axis % n_cols) != 0)
    
    elif not has_rows and has_cols and not has_axis:
        n_rows = 1
        n_axis = n_rows * n_cols
    
    elif has_rows and not has_cols and not has_axis:
        n_cols = 1
        n_axis = n_rows * n_cols
        
    elif not has_rows and not has_cols and not has_axis:
        n_rows = 1
        n_cols = 1
        n_axis = 1
    
    
    assert n_axis <= n_rows * n_cols
    
    return n_rows, n_cols, n_axis


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
    caption=None,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    xformatter: Optional[Callable[[float, float], str]] = None,
    yformatter: Optional[Callable[[float, float], str]] = None,
    left=None,
    top=None,
    bottom=None,
    right=None,
    wspace=None,
    hspace=None,
    subplot=None,
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

    if subplot is not None:
        ax = plt.gca()
        plt.subplot(*subplot)

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

    if caption is not None:
        _caption(caption)

    subplot_kwargs = _dict_of_optionals(
        left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace
    )

    if subplot_kwargs:
        plt.subplots_adjust(**subplot_kwargs)

    if legend is not None and legend is not False:
        if legend is True:
            plt.legend(loc="best")

        elif isinstance(legend, str):
            plt.legend(loc=legend)

        else:
            plt.legend(**legend)

    if subplot is not None:
        plt.sca(ax)

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
                b, which, axis = spec, "major", "both"

            elif isinstance(spec, str):
                b, which, axis = True, "major", spec

            elif isinstance(spec, tuple) and len(spec) == 2:
                b, which, axis = True, spec[0], spec[1]

            elif isinstance(spec, tuple):
                b, which, axis = spec

            else:
                raise RuntimeError()

            plt.grid(b, which, axis)

    if axis is not None and axis is not True:
        if axis is False:
            axis = "off"

        plt.axis(axis)

    # restore the previous axis
    if prev_ax is not None:
            plt.sca(prev_ax)


def diagonal(**kwargs):
    """Draw a diagonal line in the current axis."""
    import matplotlib.pyplot as plt

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    vmin = max(xmin, ymin)
    vmax = min(xmax, ymax)

    plt.plot([vmin, vmax], [vmin, vmax], **kwargs)


def qlineplot(*, x, y, hue, data, ci=0.95):
    """Plot  median as line, quantiles as shading.
    """
    import matplotlib.pyplot as plt

    agg_data = data.groupby([x, hue])[y].quantile([1 - ci, 0.5, ci]).unstack()
    hue_values = data[hue].unique()

    for color, hue_value in colorize(hue_values):
        subset = agg_data.xs(hue_value, level=hue)
        plt.fill_between(subset.index, subset.iloc[:, 0], subset.iloc[:, 2], alpha=0.2)

    for color, hue_value in colorize(hue_values):
        subset = agg_data.xs(hue_value, level=hue)
        plt.plot(subset.index, subset.iloc[:, 1], label=hue_value, marker=".")

    plt.legend(loc="best")
    plt.xlabel(x)
    plt.ylabel(y)


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


def caption(s, size=13, strip=True):
    """Add captions to matplotlib graphs."""
    import matplotlib.pyplot as plt

    if strip:
        s = s.splitlines()
        s = (i.strip() for i in s)
        s = (i for i in s if i)
        s = " ".join(s)

    plt.figtext(0.5, 0, s, wrap=True, size=size, va="bottom", ha="center")


_caption = caption


def axtext(*args, **kwargs):
    """Add a text in axes coordinates (similar ``figtext``).

    Usage::

        axtext(0, 0, 'text')

    """
    import matplotlib.pyplot as plt

    kwargs.update(transform=plt.gca().transAxes)
    plt.text(*args, **kwargs)


def _prepare_xy(x, y, data=None, transform_x=None, transform_y=None, skip_nan=True):
    if data is not None:
        x = data[x]
        y = data[y]

    x, y = _optional_skip_nan(x, y, skip_nan=skip_nan)

    if transform_x is not None:
        x = transform_x(x)

    if transform_y is not None:
        y = transform_y(y)

    return x, y


def _find_changes(v):
    import numpy as np

    (changes,) = np.nonzero(np.diff(v))
    changes = changes + 1
    return changes


def _optional_skip_nan(x, y, skip_nan=True):
    import numpy as np

    if not skip_nan:
        return x, y

    s = np.isfinite(y)
    return x[s], y[s]


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


def as_frame(**kwargs):
    import pandas as pd

    return pd.DataFrame().assign(**kwargs)


def setdefaultattr(obj, name, value):
    """``dict.setdefault`` for attributes"""
    if not hasattr(obj, name):
        setattr(obj, name, value)

    return getattr(obj, name)


# keep for backwards compat
def sapply(func, obj, sequences=(tuple,), mappings=(dict,)):
    return smap(func, obj, sequences=sequences, mappings=mappings)


def szip(
    iterable_of_objects, sequences=(tuple,), mappings=(dict,), return_schema=False
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

    return target if return_schema is False else (target, schema)


def flatten_with_index(obj, sequences=(tuple,), mappings=(dict,)):
    counter = iter(it.count())
    flat = []

    def _build(item):
        flat.append(item)
        return next(counter)

    index = smap(_build, obj, sequences=sequences, mappings=mappings)
    return index, flat


def unflatten(index, obj, sequences=(tuple,), mappings=(dict,)):
    obj = list(obj)
    return smap(lambda idx: obj[idx], index, sequences=sequences, mappings=mappings)


def smap(func, arg, *args, sequences=(tuple,), mappings=(dict,)):
    """A structured version of map.

    The structure is taken from the first arguments.
    """
    return _smap(func, arg, *args, path="$", sequences=sequences, mappings=mappings)


def _smap(func, arg, *args, path, sequences=(tuple,), mappings=(dict,)):
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


class SApplyError(Exception):
    pass


def json_numpy_default(obj):
    """A default implementation for ``json.dump`` that deals with numpy datatypes.
    """
    import numpy as np

    int_types = (
        np.int0,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint0,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    )

    float_types = (np.float16, np.float32, np.float64, np.float128)

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


def timed(tag=None, level=logging.INFO):
    """Time a codeblock and log the result.

    Usage::

        with timed():
            long_running_operation()

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
class _TimedContext(object):
    def __init__(self, logger, message, level):
        self.logger = logger
        self.message = message
        self.level = level

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.logger.log(self.level, self.message, end - self.start)


def _get_caller_logger(depth=2):
    stack = inspect.stack()

    if depth >= len(stack):  # pragma: no cover
        return logging.getLogger(__name__)

    # NOTE: python2 returns raw tuples, index 0 is the frame
    frame = stack[depth][0]
    name = frame.f_globals.get("__name__")
    return logging.getLogger(name)


def find_categorical_columns(df):
    """Find all categorical columns in the given dataframe.
    """
    import pandas.api.types as pd_types

    return [k for k, dtype in df.dtypes.items() if pd_types.is_categorical_dtype(dtype)]


def filter_low_frequency_categories(
    columns=None, min_frequency=0.02, other_category=None, n_max=None
):
    """Build a transformer to filter low frequency categories.

    Usage::

        pipeline = build_pipeline[
            categories=filter_low_frequency_categories(),
            predict=lgb.LGBMClassifier(),
        )

    """
    if columns is not None and not isinstance(columns, (list, tuple)):
        columns = [columns]

    return FilterLowFrequencyTransfomer(columns, min_frequency, other_category, n_max)


class FilterLowFrequencyTransfomer(BaseEstimator, TransformerMixin):
    def __init__(
        self, columns=None, min_frequency=0.02, other_category="other", n_max=None
    ):
        self.columns = columns
        self.min_frequency = min_frequency
        self.other_category = other_category
        self.n_max = n_max

        self._columns = columns
        self._to_keep = {}

    def fit(self, df, y=None):
        if self._columns is None:
            self._columns = find_categorical_columns(df)

        for col in self._columns:
            try:
                to_keep = find_high_frequency_categories(
                    df[col],
                    min_frequency=self._get("min_frequency", col),
                    n_max=self._get("n_max", col),
                )

            except Exception as e:
                raise RuntimeError(
                    f"cannot determine high frequency categories for {col} due to {e}"
                )

            self._to_keep[col] = to_keep

        return self

    def transform(self, df, y=None):
        for col in self._columns:
            df = df.assign(
                **{
                    col: fix_categories(
                        df[col],
                        self._to_keep[col],
                        other_category=self._get("other_category", col),
                    )
                }
            )

        return df

    def _get(self, key, col):
        var = getattr(self, key)
        return var[col] if isinstance(var, dict) else var


def make_pipeline(**kwargs):
    """Build a pipeline from named steps.

    The order of the keyword arguments is retained. Note, this functionality
    requires python ``>= 3.6``.

    Usage::

        pipeline = make_pipeline(
            transform=...,
            predict=...,
        )

    """
    import sklearn.pipeline as sk_pipeline

    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("pipeline factory requires deterministic kwarg order")

    return sk_pipeline.Pipeline(list(kwargs.items()))


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.columns_ = columns
        self.levels_ = collections.OrderedDict()

    def fit(self, x, y=None):
        if self.columns_ is None:
            self.columns_ = find_categorical_columns(x)

        for col in self.columns_:
            try:
                self.levels_[col] = multi_type_sorted(x[col].unique())

            except Exception as e:
                raise RuntimeError(f"cannot fit {col}") from e

        return self

    def transform(self, x, y=None):
        for col in self.columns_:
            try:
                assignments = {}
                for level in self.levels_[col]:
                    assignments[f"{col}_{level}"] = (x[col] == level).astype(float)

                x = x.drop([col], axis=1).assign(**assignments)

            except Exception as e:
                raise RuntimeError(f"cannot transform {col}") from e

        return x


def waterfall(
    obj,
    col=None,
    base=None,
    total=False,
    end_annot=None,
    end_fmt=".g",
    annot=False,
    fmt="+.2g",
    cmap="coolwarm",
    xmin=0,
    total_kwargs=None,
    annot_kwargs=None,
    **kwargs,
):
    """Plot a waterfall chart.

    Usage::

        series.pipe(waterfall, annot='top', fmt='+.1f', total=True)

    """
    import matplotlib.pyplot as plt
    import numpy as np

    if len(obj.shape) == 2 and col is None:
        raise ValueError("need a column with 2d objects")

    if col is not None:
        top = obj[col] if not callable(col) else col(obj)

    else:
        top = obj

    if base is not None:
        bottom = obj[base] if not callable(base) else base(obj)

    else:
        bottom = top.shift(1).fillna(0)

    if annot is True:
        annot = "top"

    if total_kwargs is None:
        total_kwargs = {}

    if annot_kwargs is None:
        annot_kwargs = {}

    if end_annot is None:
        end_annot = annot is not False

    total_kwargs = {"color": (0.5, 0.75, 0.5), **total_kwargs}

    if annot == "top":
        annot_kwargs = {"va": "bottom", "ha": "center", **annot_kwargs}
        annot_y = np.maximum(top, bottom)
        total_y = max(top.iloc[-1], 0)

    elif annot == "bottom":
        annot_kwargs = {"va": "bottom", "ha": "center", **annot_kwargs}
        annot_y = np.minimum(top, bottom)
        total_y = min(top.iloc[-1], 0)

    elif annot == "center":
        annot_kwargs = {"va": "center", "ha": "center", **annot_kwargs}
        annot_y = 0.5 * (top + bottom)
        total_y = 0.5 * top.iloc[-1]

    elif annot is not False:
        raise ValueError(f"Cannot annotate with {annot}")

    height = top - bottom

    kwargs = {"color": colormap(height, cmap=cmap, center=True), **kwargs}
    plt.bar(xmin + np.arange(len(height)), height, bottom=bottom, **kwargs)

    if annot is not False:
        for x, y, v in zip(it.count(xmin), annot_y, height):
            if x == xmin:
                continue

            plt.text(x, y, ("%" + fmt) % v, **annot_kwargs)

    if end_annot is not False:
        plt.text(xmin, annot_y.iloc[0], ("%" + end_fmt) % top.iloc[0], **annot_kwargs)

        if total:
            plt.text(
                xmin + len(annot_y),
                total_y,
                ("%" + end_fmt) % top.iloc[-1],
                **annot_kwargs,
            )

    for idx, p in zip(it.count(xmin), bottom):
        if idx == xmin:
            continue

        plt.plot([idx - 1 - 0.4, idx + 0.4], [p, p], ls="--", color="0.5")

    plt.xticks(xmin + np.arange(len(height)), list(height.index))

    if total:
        plt.bar([xmin + len(bottom)], [top.iloc[-1]], **total_kwargs)
        plt.plot(
            [xmin + len(bottom) - 1 - 0.4, xmin + len(bottom) + 0.4],
            [top.iloc[-1], top.iloc[-1]],
            ls="--",
            color="0.5",
        )


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


def bar(s, cmap="viridis", color=None, norm=None, orientation="vertical"):
    import matplotlib.colors
    import matplotlib.pyplot as plt

    if norm is None:
        norm = matplotlib.colors.NoNorm()

    if color is None:
        color = colormap(s, cmap=cmap, norm=norm)

    indices = range(len(s))

    if orientation == "vertical":
        plt.bar(indices, s, color=color)
        plt.xticks(indices, s.index)

    else:
        plt.barh(indices, s, color=color)
        plt.yticks(indices, s.index)


# TODO: make sureit can be called with a dataframe
def qplot(
    x=None,
    y=None,
    data=None,
    alpha=1.0,
    fill_alpha=0.8,
    color=None,
    ax=None,
    **line_kwargs,
):
    import numpy as np
    import matplotlib.pyplot as plt

    if y is None and x is not None:
        x, y = y, x

    if y is None:
        raise ValueError("need data to plot")

    if isinstance(y, tuple) or (isinstance(y, np.ndarray) and y.ndim == 2):
        y = tuple(y)

    else:
        y = (y,)

    if data is not None:
        y = tuple(data[c] for c in y)

    # TODO: use index if data a dataframe
    if x is None:
        x = np.arange(len(y[0]))

    elif data is not None:
        x = data[x]

    if ax is None:
        ax = plt.gca()

    if color is None:
        color = ax._get_lines.get_next_color()

    n = len(y) // 2
    fill_alpha = (1 / n) if fill_alpha is None else (fill_alpha / n)

    for i in range(n):
        plt.fill_between(x, y[i], y[-(i + 1)], alpha=fill_alpha * alpha, color=color)

    if len(y) % 2 == 1:
        plt.plot(x, y[n], alpha=alpha, color=color, **line_kwargs)


def expand(low, high, change=0.05):
    center = 0.5 * (low + high)
    delta = 0.5 * (high - low)
    return (center - (1 + 0.5 * change) * delta, center + (1 + 0.5 * change) * delta)


# ########################################################################## #
#                             I/O Methods                                    #
# ########################################################################## #


def magic_open(p, mode, *, compression=None, atomic=False):
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

    for inst in list(tqdm.tqdm._instances):
        inst.close()


# ###################################################################### #
# #                                                                    # #
# #                 Deterministic Random Number Generation             # #
# #                                                                    # #
# ###################################################################### #

maximum_15_digit_hex = float(0xFFF_FFFF_FFFF_FFFF)
max_32_bit_integer = 0xFFFF_FFFF


def sha1(obj):
    """Create a hash for a json-encode-able object
    """
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
    """Return a seed usable by numpy.
    """
    return [randrange((obj, i), max_32_bit_integer) for i in range(10)]


def tf_seed(obj):
    """Return a seed usable by tensorflow.
    """
    return randrange(obj, max_32_bit_integer)


def std_seed(obj):
    """Return a seed usable by python random module.
    """
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
    """Return the timedelta relative to weekstart for the datetime given in ``s``.
    """
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
