"""Utilities to build networks with a fluent interface"""

from typing import Dict, List

import torch

from chmp.ds import prod


def make_net(*inputs):
    """Define a neural network with given input shapes."""
    return NetBuilder(
        None,
        None,
        in_shapes=_shapes(inputs),
        out_shapes=_shapes(inputs),
    )


def register(func):
    NetBuilder._transforms[func.__name__] = func
    setattr(NetBuilder, func.__name__, func)
    setattr(make_net, func.__name__, func)

    doc = ["A fluid neural network builder", ""]
    for func in NetBuilder._transforms.values():
        if func.__doc__ is None:
            func_doc = ""

        else:
            func_doc = func.__doc__.splitlines()[0].strip()

        doc += [f"- {func.__name__}: {func_doc}"]

    doc += [
        "",
        "Call help(make.{func}) for more details, e.g., help(make_net.linear).",
        "",
        "Example:",
        "",
        "    make_mlp(20).linear(32).relu().linear(1)",
    ]

    make_net.__doc__ = "\n".join(doc)

    return func


make_net.register = register


class NetBuilder:
    _transforms = {}

    def __init__(self, decorator, obj, in_shapes, out_shapes):
        self.decorator = decorator
        self.obj = obj

        self.in_shapes = in_shapes
        self.out_shapes = out_shapes

        self.in_size = _size(*in_shapes)
        self.out_size = _size(*out_shapes)

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call unbuilt net, call .build() first")

    def _is_vector_output(self):
        return self.out_shapes == ((self.out_size,),)

    def _assert_single_output(self, ctx):
        if len(self.out_shapes) != 1:
            raise RuntimeError(f"Cannot call {ctx} with multiple outputs")

    def _assert_is_vector_output(self, ctx):
        if not self._is_vector_output():
            raise RuntimeError(
                f"Cannot call {ctx} on non-vector output, call flatten() first"
            )


@make_net.register
def build(net):
    """Build the network"""
    if net.obj is None:
        raise ValueError("Cannot build mlp without transforms")

    elif isinstance(net.obj, list):
        res = torch.nn.Sequential(*net.obj)

    else:
        res = net.obj

    if net.decorator is not None:
        res = net.decorator(res)

    return res


def chain(net, transform, out_shapes):
    if net.obj is None:
        obj = transform

    elif isinstance(net.obj, list):
        obj = [*net.obj, transform]

    else:
        obj = [net.obj, transform]

    return type(net)(
        net.decorator,
        obj,
        in_shapes=net.in_shapes,
        out_shapes=out_shapes,
    )


@make_net.register
def decorate(net, decorate, in_shapes):
    if net.decorator is not None:
        obj = build(net)

    else:
        obj = net.obj

    out_shapes = net.out_shapes if obj is not None else in_shapes

    return type(net)(decorate, obj, in_shapes=in_shapes, out_shapes=out_shapes)


@make_net.register
def pipe(net, func, *args, **kwargs):
    """Call a function on the current builder"""
    return func(net, *args, **kwargs)


@make_net.register
def repeat(net, n, func, *args, **kwargs):
    """Repeatedly call a function on the current builder"""
    for _ in range(n):
        net = func(net, *args, **kwargs)

    return net


@make_net.register
def flatten(net):
    """Append a layer to flatten the current output of the builder"""
    if net.out_shapes == ((net.out_size,),):
        return net

    if len(net.out_shapes) == 1:
        return chain(net, _FlatCat1(), out_shapes=((net.out_size,),))

    elif len(net.out_shapes) == 2:
        return decorate(net, _FlatCat2, in_shapes=((net.out_size,),))

    elif len(net.out_shapes) == 3:
        return decorate(net, _FlatCat3, in_shapes=((net.out_size,),))

    else:
        raise NotImplementedError()


@make_net.register
def linear(net, out_features, *, bias=True):
    """Append a linear layer"""
    out_size = _size(out_features)

    this = net if net._is_vector_output() else net.flatten()
    this = chain(
        this,
        torch.nn.Linear(net.out_size, out_size, bias=bias),
        out_shapes=((out_size,),),
    )
    this = this.reshape(out_features)
    return this


@make_net.register
def linears(net, *sizes, activation="relu", bias=True):
    """Append multiple linear layers with the given activation function"""
    for out_features in sizes:
        net = net.linear(out_features, bias=bias)
        if activation is not None:
            net = getattr(net, activation)()

    return net


@make_net.register
def relu(net):
    """Append a ReLU activation layer"""
    net._assert_single_output("relu")
    return chain(net, torch.nn.ReLU(), out_shapes=net.out_shapes)


@make_net.register
def sigmoid(net):
    """Append a sigmoid activation layer"""
    net._assert_single_output("sigmoid")
    return chain(net, torch.nn.Sigmoid(), out_shapes=net.out_shapes)


@make_net.register
def softplus(net):
    """Append a softplus activation layer"""
    net._assert_single_output("softplus")
    return chain(net, torch.nn.Softplus(), out_shapes=net.out_shapes)


@make_net.register
def tanh(net):
    """Append a tanh activation layer"""
    net._assert_single_output("tanh")
    return chain(net, torch.nn.Tanh(), out_shapes=net.out_shapes)


@make_net.register
def squareplus(net, alpha=1.0):
    """Append a squareplus activation layer

    Source: https://twitter.com/jon_barron/status/1387167648669048833
    """
    net._assert_single_output("squareplus")
    return chain(net, SquarePlus(alpha), out_shapes=net.out_shapes)


@make_net.register
def call(net, func_name, *args, **kwargs):
    """Call a builder function by name (e.g., an activation)."""
    return getattr(net, func_name)(*args, **kwargs)


@make_net.register
def reshape(net, shape):
    """Reshape the current output of the network (no batch dim should be passed)"""
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)

    if _size(shape) != net.out_size:
        raise RuntimeError("Reshape cannot change number of elements")

    if net.out_shapes == (shape,):
        return net

    return chain(net, Reshape((-1, *shape)), out_shapes=(shape,))


def _shapes(shapes):
    return tuple(_shape(s) for s in shapes)


def _shape(shape):
    if isinstance(shape, int):
        return (shape,)

    return tuple(shape)


def _size(*shapes):
    return sum(prod(_shape(s)) for s in shapes)


class _FlatCat1(torch.nn.Module):
    def forward(self, a):
        return a.reshape(a.shape[0], -1)


class _FlatCat2(torch.nn.Module):
    def __init__(self, then):
        super().__init__()
        self.then = then

    def forward(self, a, b):
        x = torch.cat((a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)), 1)
        return self.then(x)


class _FlatCat3(torch.nn.Module):
    def __init__(self, then):
        super().__init__()
        self.then = then

    def forward(self, a, b, c):
        x = torch.cat(
            (
                a.reshape(a.shape[0], -1),
                b.reshape(b.shape[0], -1),
                c.reshape(b.shape[0], -1),
            ),
            1,
        )
        return self.then(x)


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.reshape(self.shape)


class SquarePlus(torch.nn.Module):
    """Squareplus activation

    Source: https://twitter.com/jon_barron/status/1387167648669048833
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha2 = alpha ** 2.0

    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x ** 2.0 + self.alpha2))


class ColFillna(torch.nn.Module):
    fill_values: Dict[str, float]

    def __init__(self, fill_values):
        super().__init__()
        self.fill_values = fill_values
        self._cols = set(fill_values)

    def forward(self, x: Dict[str, torch.Tensor]):
        res = {}

        for key in x:
            if key in self.fill_values:
                res[key] = torch.nan_to_num(x[key], self.fill_values[key])

            else:
                res[key] = x[key]

        return res


class ColToDense(torch.nn.Module):
    columns: List[str]

    def __init__(self, columns):
        super().__init__()
        self.columns = columns

    def forward(self, x: Dict[str, torch.Tensor]):
        res = []

        for col in self.columns:
            res.append(x[col])

        return torch.cat(res, 1)


class ColNanIndicator(torch.nn.Module):
    columns: Dict[str, str]

    def __init__(self, columns, suffix="_nan"):
        super().__init__()
        self.columns = {col: f"{col}{suffix}" for col in columns}

    def forward(self, x: Dict[str, torch.Tensor]):
        res = {}

        for key in x:
            res[key] = x[key]
            if key in self.columns:
                res[self.columns[key]] = 1.0 - torch.isfinite(x[key]).float()
