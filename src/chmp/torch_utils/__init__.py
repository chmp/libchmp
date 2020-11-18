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


def batched_n2n(
    func,
    dtype=None,
    returns=None,
    device=None,
    sequences=default_sequences,
    mappings=default_mappings,
    arrays=default_arrays,
    tensors=default_tensors,
    batch_size=64,
):
    """Wraper to call a torch function batch-wise with numpy args and results

    This function behaves similar to :func:`n2n`, but only supports function
    arguments.

    Usage::

        pred = batched_n2n(model, batch_size=128)(x)

    """
    n2n_func = n2n(
        func,
        dtype=dtype,
        device=device,
        sequences=sequences,
        mappings=mappings,
        arrays=arrays,
        tensors=tensors,
    )

    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        dl = make_data_loader(
            NumpyDataset((args, kwargs)), mode="predict", batch_size=batch_size
        )

        with torch.no_grad():
            return szip(
                (n2n_func(*args, **kwargs) for (args, kwargs) in dl),
                combine=np.concatenate,
                sequences=default_sequences,
                mappings=default_mappings,
            )

    return wrapper


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
    """A PyTorch datast composed out of structured numpy array.

    :param data:
        the (structured) data. Nones are returned as is-is.
    :param dtype:
        if given a (structured) dtype to apply to the data
    :param filter:
         an optional boolean mask indicating which items are available
    :param sequences:
        see ``chmp.ds.smap``
    :param mappings:
        see ``chmp.ds.smap``
    """

    def __init__(
        self,
        data,
        dtype=None,
        sequences=default_sequences,
        mappings=default_mappings,
        filter=None,
    ):
        self.data = data
        self.dtype = copy_structure(data, dtype, mappings=mappings, sequences=sequences)
        self.length = self._guess_length()
        self.sequences = sequences
        self.mappings = mappings

        if filter is not None:
            (self.valid_idx,) = np.nonzero(filter)

        else:
            self.valid_idx = np.arange(self.length)

    def filter(self, func):
        """Evaluate a filter on the full data set and set the filter of this dataset"""
        return NumpyDataset(
            self.data,
            dtype=self.dtype,
            sequences=self.sequences,
            mappings=self.mappings,
            filter=func(self),
        )

    def _guess_length(self):
        candidates = set()

        def _add_len(obj):
            if obj is not None:
                candidates.add(len(obj))

        smap(_add_len, self.data)

        if len(candidates) != 1:
            raise ValueError(f"Arrays with different lengths: {candidates}")

        (length,) = candidates
        return length

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        def _get(item, dtype):
            if item is None:
                return None

            return np.asarray(item[self.valid_idx[idx]], dtype=dtype)

        return smap(
            _get,
            self.data,
            self.dtype,
            sequences=self.sequences,
            mappings=self.mappings,
        )


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


class AutoGradient:
    def __init__(self):
        pass

    def __call__(self, compute_loss):
        loss = compute_loss()
        loss.backward()
        return loss, torch.tensor(0.0, device=loss.device)


class ESGradient:
    """Estimate the gradient of a function using Evolution Strategies

    The gradient will be assigned to the ``grad`` property of the parameters.
    This way any PyTorch optimizer can be used. As the tensors are manipulated
    in-place, they must not require gradients. For modules or tensors call
    ``requires_grad_(False)`` before using ``ESGradient``. The return value will
    be the mean and std of the loss.

    Usage::

        grad_fn = ESGradient(model.parameters())
        optimizer = torch.optim.Adam(model.parameters())

        # ...
        optimizer.zero_grad()
        grad_fn(lambda: compute_loss(model))
        optimizer.step()

    :param parameters: the parameters as an iterable :param n_samples: the
        number of samples with which to estimate the gradient :param scale: the
        scale of the perturbation to use. Can be passed as a list with the same
        length as parameters to give different scales for each parameter.
    """

    def __init__(self, parameters, *, n_samples=50, scale=0.1):
        self.parameters = list(parameters)
        self.n_samples = n_samples
        self.scale = scale

    def __call__(self, compute_loss):
        return _add_es_grad(
            self.parameters,
            compute_loss,
            n_samples=self.n_samples,
            scale=self.scale,
        )


def _add_es_grad(params, compute_loss, *, n_samples, scale):
    assert n_samples % 2 == 0

    params = list(params)

    if not isinstance(scale, (list, tuple)):
        scale = [scale] * len(params)

    assert len(params) == len(scale)

    def _step(compute_param):
        for p, c, s in zip(params, centers, scale):
            p.copy_(compute_param(p, c, s))

        loss = compute_loss()

        for p, c, s in zip(params, centers, scale):
            p.grad.add_(loss / (n_samples * s) * (p - c))

        return [torch.as_tensor(loss)]

    with torch.no_grad():
        params = [p for p in params]
        centers = [p.clone() for p in params]

        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        losses = []

        for _ in range(n_samples // 2):
            # compute the positive sample
            losses += _step(lambda _, c, s: c + s * torch.randn(c.shape))

            # compute the negative sample
            losses += _step(lambda p, c, _: c - (p - c))

        for p, c in zip(params, centers):
            p.copy_(c)

        losses = torch.stack(losses)
        return torch.mean(losses), torch.std(losses)


def update_moving_average(alpha, average, value):
    """Update iterables of tensors by an exponentially moving average.

    If ``average`` and ``value`` are passed as module parameters, this function
    can be used to make one module the moving average of the other module::

        target_value_function = copy.copy(value_function)
        target_value_function.requires_grad_(False)

        # ...

        update_moving_average(
            0.9,
            target_value_function.parameters(),
            value_function.parameters(),
        )

    """
    with torch.no_grad():
        for a, v in zip(average, value):
            a.mul_(alpha)
            a.add_((1 - alpha) * v)
