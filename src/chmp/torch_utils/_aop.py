"""A minimalist aspect orientied programming lib
"""
import collections
import contextlib
import contextvars
import functools as ft
import inspect


_aspects = contextvars.ContextVar("aspects")
_modified_root = contextvars.ContextVar("modified_root")


@contextlib.contextmanager
def modify(root):
    """Customizing the root by adding new aspects."""
    root, aspects = _unwrap_root(root)

    @ft.wraps(root)
    def wrapper(*args, **kwargs):
        return run_with_aspects(root, wrapper.aspects, *args, **kwargs)

    wrapper.aspects = aspects

    if is_joinpoint(root):
        wrapper._joinpoint_ = root._joinpoint_

    token = _modified_root.set(wrapper)
    try:
        yield wrapper

    finally:
        _modified_root.reset(token)


def run_with_aspects(root, aspects, /, *args, **kwargs):
    try:
        token = _aspects.set(aspects.copy())
        return root(*args, **kwargs)

    finally:
        _aspects.reset(token)


def joinpoint(key):
    """Mark a function as a possible joinpoint that can be customized.

    Usage::

        @joinpoint("my_joinpoint")
        def my_joinpoint():
            ...

    """
    assert isinstance(key, str)

    def decorator(func):
        def root_aspect(proceed, /, *args, **kwargs):
            return func(*args, **kwargs)

        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            aspects = [*_get_active_aspects(key), root_aspect]
            return proceed(aspects, *args, **kwargs)

        wrapper._joinpoint_ = key
        return wrapper

    return decorator


def proceed(chain, /, *args, **kwargs):
    head, *tail = chain
    return head(ft.partial(proceed, tail), *args, **kwargs)


def is_joinpoint(obj):
    """Is the passed object a joinpoint?"""
    return hasattr(obj, "_joinpoint_")


def add_aspect(aspect):
    """Add an aspect to the currently root currently being modified."""
    modified_root = _modified_root.get()
    _update_aspects(modified_root.aspects, aspect)
    return aspect


def _aspect_factory(impl):
    def wrapper(point):
        def decorator(advice):
            add_aspect({point: [ft.partial(impl, advice)]})
            return advice

        return decorator

    wrapper.__name__ = impl.__name__
    wrapper.__doc__ = impl.__doc__

    return wrapper


@_aspect_factory
def before(advice, proceed, /, *args, **kwargs):
    """Run the advice before proceeding."""
    advice_args, advice_kwargs = _get_advice_args(advice, args, kwargs)
    advice(*advice_args, **advice_kwargs)
    return proceed(*args, **kwargs)


@_aspect_factory
def after(advice, proceed, /, *args, **kwargs):
    """Run the advice after proceeding, the result is passed as the first arg."""
    res = proceed(*args, **kwargs)

    advice_args, advice_kwargs = _get_advice_args(advice, (res, *args), kwargs)
    advice(*advice_args, **advice_kwargs)
    return res


@_aspect_factory
def around(advice, proceed, /, *args, **kwargs):
    """Call the advice with proceed as the first argument"""
    advice_args, advice_kwargs = _get_advice_args(advice, (proceed, *args), kwargs)
    return advice(*advice_args, **advice_kwargs)


@_aspect_factory
def replace(advice, proceed, /, *args, **kwargs):
    """Call the advice instead of proceed"""
    advice_args, advice_kwargs = _get_advice_args(advice, args, kwargs)
    return advice(*advice_args, **advice_kwargs)


def _get_advice_args(advice, args, kwargs):
    sig = inspect.signature(advice)

    accept_posargs = 0
    accept_kwargs = collections.OrderedDict()

    for name, desc in sig.parameters.items():
        if desc.kind in {desc.POSITIONAL_OR_KEYWORD, desc.POSITIONAL_ONLY}:
            accept_posargs += 1

        elif desc.kind == desc.VAR_POSITIONAL:
            accept_posargs = len(args)

        elif desc.kind == desc.KEYWORD_ONLY:
            accept_kwargs[name] = None

        elif desc.kind == desc.VAR_KEYWORD:
            for key in kwargs:
                if not key in accept_kwargs:
                    accept_kwargs[key] = None

    assert accept_posargs <= len(args)

    return args[:accept_posargs], {key: kwargs[key] for key in accept_kwargs}


def _unwrap_root(root):
    if hasattr(root, "aspects"):
        aspects = {k: list(v) for k, v in root.aspects.items()}
        root = root.__wrapped__

    else:
        aspects = {}

    return root, aspects


def _update_aspects(target, obj):
    for point, advices in _get_aspects(obj).items():
        target[point] = [*advices, *target.get(point, [])]

    return target


def _get_active_aspects(key):
    aspects = _aspects.get({})
    return aspects.get(key, [])


def _get_aspects(obj):
    if hasattr(obj, "_aspects"):
        obj = obj._aspects

    assert isinstance(obj, dict)
    return obj
