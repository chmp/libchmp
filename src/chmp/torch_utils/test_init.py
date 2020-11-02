import copy

import numpy as np
import pandas as pd
import pytest
import torch

from chmp.torch_utils import (
    t2n,
    n2t,
    t2t,
    n2n,
    make_mlp,
    ESGradient,
    update_moving_average,
)


def test_t2n_nested_structure():
    actual = t2n({"foo": torch.tensor(0), "bar": (torch.tensor(1), torch.tensor(2))})
    expected = {
        "foo": np.asarray(0),
        "bar": (
            np.asarray(1),
            np.asarray(2),
        ),
    }

    assert actual == expected


def test_t2n_decorator_example():
    """n2t can be used as a decroator"""

    @t2n
    def func(x):
        assert isinstance(x, np.ndarray)
        return x

    func(torch.randn(size=()))
    func(x=torch.randn(size=()))


def test_t2n_nested_structure():
    actual = t2n({"foo": torch.tensor(0), "bar": (torch.tensor(1), torch.tensor(2))})
    expected = {
        "foo": np.asarray(0),
        "bar": (
            np.asarray(1),
            np.asarray(2),
        ),
    }

    assert actual == expected


def test_n2t_negative_strides():
    n2t(np.arange(5)[::-1])


def test_n2t_nested_structure():
    actual = n2t({"foo": np.asarray(0), "bar": (np.asarray(1), np.asarray(2))})
    expected = {
        "foo": torch.tensor(0),
        "bar": (
            torch.tensor(1),
            torch.tensor(2),
        ),
    }

    assert actual == expected


def test_n2t_decorator_example():
    """n2t can be used as a decroator"""

    @n2t
    def func(x):
        assert torch.is_tensor(x)
        return x

    func(np.random.normal(size=()))
    func(x=np.random.normal(size=()))


def test_custom_arrays_args():
    @n2t
    def func(x):
        assert torch.is_tensor(x)
        return x

    func(pd.DataFrame(np.random.normal(size=(5, 5))))
    func(x=pd.DataFrame(np.random.normal(size=(5, 5))))


def test_n2n_example():
    @n2n
    def func(x):
        assert torch.is_tensor(x)
        return x

    res = func(np.random.normal(size=()))
    assert isinstance(res, np.ndarray)


def test_t2t_example():
    @t2t(dtype=dict(x="float32", a="float32"))
    def func(x, a):
        assert isinstance(x, np.ndarray)
        return x

    res = func(np.random.normal(size=()), np.random.normal(size=()))
    assert isinstance(res, torch.Tensor)


def test_make_mlp():
    make_mlp(5, 5)
    make_mlp(5, 5, activation=torch.nn.ReLU)
    make_mlp(5, 5, hidden=(10,), activation=torch.nn.ReLU)
    make_mlp(5, 5, hidden=10, activation=torch.nn.Softplus)


def test_esgradient_example():
    a = torch.nn.Parameter(torch.randn(1, 5))
    b = torch.nn.Parameter(torch.randn(1, 5))

    optim = torch.optim.SGD([a, b], lr=1.0)
    grad_fn = ESGradient([a, b])

    optim.zero_grad()
    grad_fn(lambda: (a * b).sum())
    optim.step()


def test_esgradient_example_different_scales():
    a = torch.nn.Parameter(torch.randn(1, 5))
    b = torch.nn.Parameter(torch.randn(1, 5))

    optim = torch.optim.SGD([a, b], lr=1.0)
    grad_fn = ESGradient([a, b], scale=[0.5, 1.0])

    optim.zero_grad()
    grad_fn(lambda: (a * b).sum())
    optim.step()


def test_update_moving_average():
    m = torch.nn.Linear(5, 5)

    m2 = copy.deepcopy(m)
    m2.requires_grad_(False)

    update_moving_average(0.9, m2.parameters(), m.parameters())
