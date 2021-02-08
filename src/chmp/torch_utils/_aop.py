import contextvars
import contextlib
import functools as ft


class System:
    def __init__(self, root, aspects=None, prototype=None):
        if aspects is None:
            aspects = {}

        if prototype is None:
            prototype = {}

        self.root = root
        self.aspects = aspects
        self.prototype = {}

    def __call__(self, *args, **kwargs):
        with Context(self), Context.call(self.root):
            return proceed(*args, **kwargs)

    def copy(self):
        return type(self)(
            self.root, _copy_aspects(self.aspects), _copy_aspects(self.prototype)
        )

    def reset(self):
        self.aspects = _copy_aspects(self.prototype)


def _copy_aspects(aspects):
    return {k: list(v) for k, v in aspects.items()}


def proceed(*args, **kwargs):
    return Context.proceed(*args, **kwargs)


def replace(system, point, *, prototype=False):
    return _decorator_impl(system, point, prototype, lambda curr, prev: [curr])


def decorate(system, point, *, prototype=False):
    return _decorator_impl(system, point, prototype, lambda curr, prev: [curr, *prev])


def before(system, point, *, prototype=False):
    return _decorator_impl(
        system, point, prototype, lambda curr, prev: [curr, *prev], _before_wrapper
    )


def after(system, point, *, prototype=False):
    return _decorator_impl(
        system, point, prototype, lambda curr, prev: [curr, *prev], _after_wrapper
    )


def add_aspect(system, aspect, *, prototype=False):
    targets = _targets(system, prototype=prototype)

    for point, advice in aspect._aspects.items():
        key = _key(point)
        for target in targets:
            target[key] = [advice, *target.get(key, [])]

    return aspect


class joinpoint:
    def __init__(self, key):
        if not isinstance(key, str):
            key = f"{key.__module__}.{key.__name__}"

        self.key = key

    def __call__(self, *args, **kwargs):
        with Context.call(self.key):
            return proceed(*args, **kwargs)


class Context:
    _active = contextvars.ContextVar("_active", default=None)

    def __init__(self, system):
        self.system = system
        self.stack = []

    @classmethod
    def call(cls, point):
        return _context_call_impl(cls, point)

    @classmethod
    def proceed(*args, **kwargs):
        cls, *args = args

        # TODO: clean this up
        ctx = cls._active.get()

        stack_idx = len(ctx.stack) - 1
        key, chain_idx = ctx.stack[stack_idx]
        ctx.stack[stack_idx] = (key, chain_idx + 1)

        try:
            try:
                func_stack = ctx.system.aspects[key]
            except KeyError as err:
                raise RuntimeError(f"No implementation for {key} found") from err

            try:
                func = func_stack[chain_idx]

            except IndexError as err:
                raise RuntimeError(
                    f"Internal error for {key}: proceed called in tail"
                ) from err

            return func(*args, **kwargs)

        finally:
            ctx.stack[stack_idx] = (key, chain_idx)

    def __enter__(self):
        assert self._active.get() is None
        self._active.set(self)

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._active.get() is self
        self._active.set(None)


@contextlib.contextmanager
def _context_call_impl(cls, point):
    inst = cls._active.get()
    assert inst is not None

    key = _key(point)
    inst.stack.append((key, 0))
    try:
        yield

    finally:
        inst.stack.pop()


def _decorator_impl(system, point, prototype, impl, wrapper=None):
    key = _key(point)
    targets = _targets(system, prototype=prototype)

    def decorator(func):
        if wrapper is not None:
            func = wrapper(func)

        for target in targets:
            target[key] = impl(func, target.get(key, []))
        return func

    return decorator


def _key(obj):
    if hasattr(obj, "key"):
        return obj.key

    return obj


def _targets(system, prototype=False):
    if prototype:
        return [system.aspects, system.prototype]

    else:
        return [system.aspects]


def _before_wrapper(func):
    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        return proceed(*args, **kwargs)

    wrapper._aspect_hint = "before"

    return wrapper


def _after_wrapper(func):
    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        res = proceed(*args, **kwargs)
        func(*args, **kwargs)
        return res

    wrapper._aspect_hint = "after"

    return wrapper
