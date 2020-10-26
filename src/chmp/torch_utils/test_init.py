import numpy as np
import pandas as pd
import pytest
import torch

from chmp.torch_utils import t2n, n2t, t2t, n2n


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


def test_n2t_dataframes_default():
    """n2t does not handle DataFrames per default."""

    @n2t
    def func(x):
        assert torch.is_tensor(x)
        return x

    with pytest.raises(AssertionError):
        func(pd.DataFrame(np.random.normal(size=(5, 5))))


def test_custom_arrays_args():
    """n2t can be customized to handle DataFrames"""

    @n2t(arrays=(np.ndarray, pd.DataFrame))
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
    @t2t(dtype=(("float32", "float32"), {}))
    def func(x, a):
        assert isinstance(x, np.ndarray)
        return x

    res = func(np.random.normal(size=()), np.random.normal(size=()))
    assert isinstance(res, torch.Tensor)
