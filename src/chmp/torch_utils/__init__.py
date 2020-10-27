"""Helper to construct models with pytorch."""
import collections
import enum
import functools as ft
import itertools as it
import operator as op

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from typing import Union, Sequence

from chmp.ds import (
    smap,
    szip,
    transform_args,
    flatten_with_index,
    unflatten,
    copy_structure,
    default_sequences,
    default_mappings,
    undefined,
)

try:
    import pandas as pd

except ImportError:
    _pd_arrays = ()

else:
    _pd_arrays = (pd.Series, pd.DataFrame)


default_arrays = (np.ndarray, *_pd_arrays)
default_tensors = (torch.Tensor,)

default_batch_size = 32


class fixed:
    """decorator to mark a parameter as not-optimized."""

    def __init__(self, value):
        self.value = value


class optimized:
    """Decorator to mark a parameter as optimized."""

    def __init__(self, value):
        self.value = value


def optional_parameter(arg, *, default=optimized):
    """Make sure arg is a tensor and optionally a parameter.

    Values wrapped with ``fixed`` are returned as a tensor, ``values`` wrapped
    with ``optimized``are returned as parameters. When arg is not one of
    ``fixed`` or ``optimized`` it is wrapped with ``default``.

    Usage::

        class MyModule(torch.nn.Module):
            def __init__(self, a, b):
                super().__init__()

                # per default a will be optimized during training
                self.a = optional_parameter(a, default=optimized)

                # per default B will not be optimized during training
                self.b = optional_parameter(b, default=fixed)

    """
    if isinstance(arg, fixed):
        return torch.as_tensor(arg.value)

    elif isinstance(arg, optimized):
        return torch.nn.Parameter(torch.as_tensor(arg.value))

    elif default is optimized:
        return torch.nn.Parameter(torch.as_tensor(arg))

    elif default is fixed:
        return torch.as_tensor(arg)

    else:
        raise RuntimeError()


def register_unknown_kl(type_p, type_q):
    def decorator(func):
        if has_kl(type_p, type_q):
            func.registered = False
            return func

        torch.distributions.kl.register_kl(type_p, type_q)(func)
        func.registered = True
        return func

    return decorator


def has_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY


def t2n(
    obj=undefined,
    *,
    dtype=None,
    sequences=default_sequences,
    mappings=default_mappings,
    tensors=default_tensors,
):
    """Torch to numpy."""
    if obj is undefined:
        return ft.partial(
            t2n, dtype=dtype, sequences=sequences, mappings=mappings, tensors=tensors
        )

    if not callable(obj):
        return _t2n_tensors(
            obj, dtype=dtype, sequences=sequences, mappings=mappings, tensors=tensors
        )

    @ft.wraps(obj)
    def wrapper(*args, **kwargs):
        args, kwargs = transform_args(
            obj,
            args,
            kwargs,
            _t2n_tensors,
            dtype=dtype,
            sequences=sequences,
            mappings=mappings,
            tensors=tensors,
        )
        return obj(*args, **kwargs)

    return wrapper


def _t2n_tensors(obj, *, dtype, sequences, mappings, tensors):
    dtype = copy_structure(obj, dtype, sequences=sequences, mappings=mappings)
    return smap(
        ft.partial(_t2n_scalar, tensors=tensors),
        obj,
        dtype,
        sequences=sequences,
        mappings=mappings,
    )


def _t2n_scalar(obj, dtype, tensors):
    if not isinstance(obj, tensors):
        return obj

    return np.asarray(obj.detach().cpu(), dtype=dtype)


def n2t(
    obj=undefined,
    *,
    dtype=None,
    device=None,
    sequences=default_sequences,
    mappings=default_mappings,
    arrays=default_arrays,
):
    """Numpy to torch."""
    if obj is undefined:
        return ft.partial(
            n2t,
            dtype=dtype,
            device=device,
            sequences=sequences,
            mappings=mappings,
            arrays=arrays,
        )

    if not callable(obj):
        return _n2t_tensors(
            obj,
            dtype=dtype,
            device=device,
            sequences=sequences,
            mappings=mappings,
            arrays=arrays,
        )

    @ft.wraps(obj)
    def wrapper(*args, **kwargs):
        args, kwargs = transform_args(
            obj,
            args,
            kwargs,
            _n2t_tensors,
            dtype=dtype,
            device=device,
            sequences=sequences,
            mappings=mappings,
            arrays=arrays,
        )
        return obj(*args, **kwargs)

    return wrapper


def _n2t_tensors(obj, *, dtype, device, sequences, mappings, arrays):
    dtype = copy_structure(obj, dtype, sequences=sequences, mappings=mappings)
    device = copy_structure(obj, device, sequences=sequences, mappings=mappings)

    return smap(
        ft.partial(_n2t_scalar, arrays=arrays),
        obj,
        dtype,
        device,
        sequences=sequences,
        mappings=mappings,
    )


def _n2t_scalar(obj, dtype, device, arrays):
    if not isinstance(obj, arrays):
        return obj

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if isinstance(device, str):
        device = torch.device(device)

    obj = np.asarray(obj)

    # torch cannot handle negative strides, make a copy to remove striding
    if any(s < 0 for s in obj.strides):
        obj = obj.copy()

    return torch.as_tensor(obj, dtype=dtype, device=device)


def t2t(
    func=undefined,
    *,
    dtype=None,
    returns=None,
    device=None,
    sequences=default_sequences,
    mappings=default_mappings,
    arrays=default_arrays,
    tensors=default_tensors,
):
    """Equivalent  to ``n2t(t2n(func)(*args, **kwargs)``"""
    if func is undefined:
        return ft.partial(
            t2t,
            dtype=dtype,
            returns=returns,
            device=device,
            sequences=sequences,
            mappings=mappings,
            arrays=arrays,
            tensors=tensors,
        )

    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = transform_args(
            func,
            args,
            kwargs,
            _t2n_tensors,
            dtype=dtype,
            sequences=sequences,
            mappings=mappings,
            tensors=tensors,
        )
        res = func(*args, **kwargs)
        return n2t(
            res,
            dtype=returns,
            device=device,
            sequences=sequences,
            mappings=mappings,
            arrays=arrays,
        )

    return wrapper


def n2n(
    func=undefined,
    *,
    dtype=None,
    returns=None,
    device=None,
    sequences=default_sequences,
    mappings=default_mappings,
    arrays=default_arrays,
    tensors=default_tensors,
):
    """Equivalent to ``t2n(n2t(func)(*args, **kwargs)``"""
    if func is undefined:
        return ft.partial(
            n2n,
            dtype=dtype,
            returns=returns,
            device=device,
            sequences=sequences,
            mappings=mappings,
            arrays=arrays,
            tensors=tensors,
        )

    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = transform_args(
            func,
            args,
            kwargs,
            _n2t_tensors,
            dtype=dtype,
            device=device,
            sequences=sequences,
            mappings=mappings,
            arrays=arrays,
        )
        res = func(*args, **kwargs)
        return t2n(
            res,
            dtype=returns,
            sequences=sequences,
            mappings=mappings,
            tensors=tensors,
        )

    return wrapper


def call_torch(func, arg, *args, dtype=None, device=None, batch_size=64):
    """Call a torch function with numpy arguments and numpy results."""
    args = (arg, *args)
    index, values = flatten_with_index(args)
    result_batches = []

    for start in it.count(0, batch_size):
        end = start + batch_size

        if start >= len(values[0]):
            break

        batch = unflatten(index, (val[start:end] for val in values))
        batch = n2t(batch, dtype=dtype, device=device)
        result = func(*batch)
        result = t2n(result)

        result_batches.append(result)

    result, schema = szip(result_batches, return_schema=True)
    result = smap(lambda _, r: np.concatenate(r, axis=0), schema, result)
    return result


def optimizer_step(optimizer, func, *args, **kwargs):
    optimizer.zero_grad()
    loss = func(*args, **kwargs)

    if isinstance(loss, tuple):
        loss[0].backward()

    elif isinstance(loss, dict):
        loss["loss"].backward()

    else:
        loss.backward()

    optimizer.step()

    return smap(float, loss)


def identity(x):
    return x


def linear(x, weights):
    """A linear interaction.

    :param x:
        shape ``(batch_size, in_features)``
    :param weights:
        shape ``(n_factors, in_features, out_features)``
    """
    return x @ weights


def factorized_quadratic(x, weights):
    """A factorized quadratic interaction.

    :param x:
        shape ``(batch_size, in_features)``
    :param weights:
        shape ``(n_factors, in_features, out_features)``
    """
    x = x[None, ...]
    res = (x @ weights) ** 2.0 - (x ** 2.0) @ (weights ** 2.0)
    res = res.sum(dim=0)
    return 0.5 * res


def masked_softmax(logits, mask, axis=-1, eps=1e-9):
    keep = 1.0 - mask.type(logits.dtype)
    p = keep * torch.softmax(logits * keep, axis=axis)
    p = p / (eps + p.sum(axis=axis, keepdims=True))
    return p


def find_module(root, predicate):
    """Find a (sub) module using a predicate.

    :param predicate:
        a callable with arguments ``(name, module)``.
    :returns:
        the first module for which the predicate is true or raises
        a ``RuntimeError``.
    """
    for k, v in root.named_modules():
        if predicate(k, v):
            return v

    else:
        raise RuntimeError("could not find module")


class DiagonalScaleShift(torch.nn.Module):
    """Scale and shift the inputs along each dimension independently."""

    @classmethod
    def from_data(cls, data):
        return cls(shift=data.mean(), scale=1.0 / (1e-5 + data.std()))

    def __init__(self, shift=None, scale=None):
        super().__init__()
        assert (shift is not None) or (scale is not None)

        if shift is not None:
            shift = torch.as_tensor(shift).clone()

        if scale is not None:
            scale = torch.as_tensor(scale).clone()

        if shift is None:
            shift = torch.zeros_like(scale)

        if scale is None:
            scale = torch.ones_like(shift)

        self.shift = torch.nn.Parameter(shift)
        self.scale = torch.nn.Parameter(scale)

    def forward(self, x):
        return self.scale * (x - self.shift)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def format_extra_repr(*kv_pairs):
    return ", ".join("{}={}".format(k, v) for k, v in kv_pairs)


class CallableWrapper(torch.nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def extra_repr(self):
        return format_extra_repr(("func", self.func), *self.kwargs.items())


class Do(CallableWrapper):
    """Call a function as a pure side-effect."""

    def forward(self, x, **kwargs):
        self.func(x, **kwargs, **self.kwargs)
        return x


class Lambda(CallableWrapper):
    def forward(self, *x, **kwargs):
        return self.func(*x, **kwargs, **self.kwargs)


class LocationScale(torch.nn.Module):
    def __init__(self, activation=None, eps=1e-6):
        super().__init__()

        if activation is None:
            activation = Identity()

        self.eps = eps
        self.activation = activation

    def forward(self, x):
        *_, n = x.shape
        assert (n % 2) == 0, "can only handle even number of features"

        loc = x[..., : (n // 2)]
        scale = x[..., (n // 2) :]

        loc = self.activation(loc)
        scale = self.eps + F.softplus(scale)

        return loc, scale

    def extra_repr(self):
        return f"eps={self.eps},"


# TODO: figure out how to properly place the nodes
# TODO: use linear interpolation
class LookupFunction(torch.nn.Module):
    """Helper to define a lookup function incl. its gradient.

    Usage::

        import scipy.special

        x = np.linspace(0, 10, 100).astype('float32')
        iv0 = scipy.special.iv(0, x).astype('float32')
        iv1 = scipy.special.iv(1, x).astype('float32')

        iv = LookupFunction(x.min(), x.max(), iv0, iv1)

        a = torch.linspace(0, 20, 200, requires_grad=True)
        g, = torch.autograd.grad(iv(a), a, torch.ones_like(a))

    """

    def __init__(self, input_min, input_max, forward_values, backward_values):
        super().__init__()
        self.input_min = torch.as_tensor(input_min)
        self.input_max = torch.as_tensor(input_max)
        self.forward_values = torch.as_tensor(forward_values)
        self.backward_values = torch.as_tensor(backward_values)

    def forward(self, x):
        return _LookupFunction.apply(
            x, self.input_min, self.input_max, self.forward_values, self.backward_values
        )


class _LookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, input_min, input_max, forward_values, backward_values):
        idx_max = len(forward_values) - 1
        idx_scale = idx_max / (input_max - input_min)
        idx = (idx_scale * (x - input_min)).type(torch.long)
        idx = torch.clamp(idx, 0, idx_max)

        if backward_values is not None:
            ctx.save_for_backward(backward_values[idx])

        else:
            ctx.save_for_backward(None)

        return forward_values[idx]

    @staticmethod
    def backward(ctx, grad_output):
        (backward_values,) = ctx.saved_tensors
        return grad_output * backward_values, None, None, None, None


def make_mlp(
    in_features: int,
    out_features: int,
    *,
    hidden: Union[Sequence[int], int] = (),
    hidden_activation=torch.nn.ReLU,
    activation=None,
    container=torch.nn.Sequential,
):
    if isinstance(hidden, int):
        hidden = [hidden]

    in_features = [in_features, *hidden]
    out_features = [*hidden, out_features]
    activations = len(hidden) * [hidden_activation] + [activation]

    parts = []
    for a, b, activation in zip(in_features, out_features, activations):
        parts += [torch.nn.Linear(a, b)]

        if activation is not None:
            parts += [activation()]

    return container(*parts) if len(parts) != 1 else parts[0]


def make_data_loader(dataset, mode="fit", **kwargs):
    if mode == "fit":
        default_kwargs = dict(shuffle=True, drop_last=True)

    elif mode == "predict":
        default_kwargs = dict(shuffle=False, drop_last=False)

    else:
        raise ValueError()

    kwargs = {**default_kwargs, **kwargs}
    return torch.utils.data.DataLoader(dataset, **kwargs)


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data, dtype=None):
        index, data = flatten_with_index(data)
        data = [np.asarray(v) for v in data]

        self.index = index
        self.data = data
        self.dtype = dtype
        self.length = self._guess_length()

    def _guess_length(self):
        candidates = set()

        for item in self.data:
            if item is None:
                continue

            candidates.add(len(item))

        if len(candidates) != 1:
            raise ValueError(f"Arrays with different lengths: {candidates}")

        (length,) = candidates
        return length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        res = [(item[idx] if item is not None else None) for item in self.data]
        if self.dtype is not None:
            res = [
                (np.asarray(item, dtype=self.dtype) if item is not None else None)
                for item in res
            ]

        return unflatten(self.index, res)


@register_unknown_kl(torch.distributions.LogNormal, torch.distributions.Gamma)
def kl_divergence__gamma__log_normal(p, q):
    """Compute the kl divergence with a Gamma prior and LogNormal approximation.

    Taken from C. Louizos, K. Ullrich, M. Welling "Bayesian Compression for Deep Learning"
    https://arxiv.org/abs/1705.08665
    """
    return (
        q.concentration * torch.log(q.rate)
        + torch.lgamma(q.concentration)
        - q.concentration * p.loc
        + torch.exp(p.loc + 0.5 * p.scale ** 2) / q.rate
        - 0.5 * (torch.log(p.scale ** 2.0) + 1 + np.log(2 * np.pi))
    )
