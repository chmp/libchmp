"""Utilities to build networks with a fluent interface"""

import functools as ft

import torch

from chmp.ds import prod


def make_net(*inputs):
    """Define a neural network with given input shapes.
    """
    return NetBuilder(
        None,
        None,
        in_shapes=_shapes(inputs),
        out_shapes=_shapes(inputs),
    )


def register(func):
    NetBuilder._net_transforms[func.__name__] = func
    return func


make_net.register = register


class DynamicDoc:
    def __get__(self, obj, objtype=None):
        return "foo"


make_net.__doc__ = DynamicDoc()


class NetBuilder:
    _net_transforms = {}

    def __init__(self, decorator, obj, in_shapes, out_shapes):
        self.decorator = decorator
        self.obj = obj

        self.in_shapes = in_shapes
        self.out_shapes = out_shapes

        self.in_size = _size(*in_shapes)
        self.out_size = _size(*out_shapes)

    def __getattr__(self, key):
        return ft.partial(self._net_transforms[key], self)

    def _is_vector_output(self):
        return self.out_shapes == ((self.out_size,),)

    def _assert_single_output(self, ctx):
        if len(self.out_shapes) != 1:
            raise RuntimeError(f"Cannot call {ctx} with multiple outputs")

    def _assert_is_vector_output(self, ctx):
        if not self._is_vector_output():
            raise RuntimeError(f"Cannot call {ctx} on non-vector output, call flatten() first")



@make_net.register
def build(net):
    if net.obj is None:
        raise ValueError("Cannot build mlp without transforms")

    elif isinstance(net.obj, list):
        res = torch.nn.Sequential(*net.obj)

    else:
        res = net.obj

    if net.decorator is not None:
        res = net.decorator(res)

    return res


@make_net.register
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
    return func(net, *args, **kwargs)


@make_net.register
def repeat(net, n, func, *args, **kwargs):
    for _ in range(n):
        net = func(net, *args, **kwargs)

    return net


@make_net.register
def flatten(net):
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
    out_size = _size(out_features)

    this = net if net._is_vector_output() else net.flatten()
    this = this.chain(
        torch.nn.Linear(net.out_size, out_size, bias=bias),
        out_shapes=((out_size,),),
    )
    this = this.reshape(out_features)
    return this


@make_net.register
def linears(net, *sizes, activation=None, bias=True):
    for out_features in sizes:
        net = net.linear(out_features, bias=bias)
        if activation is not None:
            net = getattr(net, activation)()

    return net


@make_net.register
def relu(net):
    net._assert_single_output("relu")
    return chain(net, torch.nn.ReLU(), out_shapes=net.out_shapes)


@make_net.register
def sigmoid(net):
    net._assert_single_output("sigmoid")
    return chain(net, torch.nn.Sigmoid(), out_shapes=net.out_shapes)


@make_net.register
def softplus(net):
    net._assert_single_output("softplus")
    return chain(net, torch.nn.Softplus(), out_shapes=net.out_shapes)


@make_net.register
def tanh(net):
    net._assert_single_output("tanh")
    return chain(net, torch.nn.Tanh(), out_shapes=net.out_shapes)


@make_net.register
def reshape(net, shape):
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)

    if _size(shape) != net.out_size:
        raise RuntimeError("Reshape cannot change number of elements")

    if net.out_shapes == (shape,):
        return net

    return net.chain(Reshape((-1, *shape)), out_shapes=(shape,))


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
