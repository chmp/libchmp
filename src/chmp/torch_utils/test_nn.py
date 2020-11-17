import numpy as np
import pandas as pd
import torch

from chmp.torch_utils import (
    factorized_quadratic,
    masked_softmax,
    linear,
    DiagonalScaleShift,
    batched_n2n,
    t2n,
    NumpyDataset,
)


def test_linear_shape():
    weights = torch.zeros(10, 5)
    assert linear(torch.zeros(20, 10), weights).shape == (20, 5)


def test_factorized_quadratic_shape():
    weights = torch.zeros(2, 10, 5)
    assert factorized_quadratic(torch.zeros(20, 10), weights).shape == (20, 5)


def test_masked_softmax():
    actual = masked_softmax(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([False, False, True]),
    )
    actual = np.asarray(actual)

    expected = np.asarray([np.exp(1), np.exp(2), 0])
    expected = expected / expected.sum()

    np.testing.assert_allclose(actual, expected)


def test_diagonal_scale_shift():
    m = DiagonalScaleShift(shift=torch.ones(10), scale=2.0 * torch.ones(10))
    assert m(torch.zeros(20, 10)).shape == (20, 10)


def test_call_torch():
    np.testing.assert_almost_equal(
        batched_n2n(lambda x: torch.sqrt(x))(np.asarray([1, 4, 9], dtype="float")),
        [1, 2, 3],
    )

    np.testing.assert_almost_equal(
        batched_n2n(lambda a, b: a + b)(
            np.asarray([1, 2, 3], dtype="float"),
            np.asarray([4, 5, 6], dtype="float"),
        ),
        [5, 7, 9],
    )


def test_call_torch_structured():
    a, b = batched_n2n(lambda t: (t[0] + t[1], t[1] - t[0]))(
        (np.asarray([1, 2, 3], dtype="float"), np.asarray([4, 5, 6], dtype="float")),
    )

    np.testing.assert_almost_equal(a, [5, 7, 9])
    np.testing.assert_almost_equal(b, [3, 3, 3])


def test_call_torch_batched():
    np.testing.assert_almost_equal(
        batched_n2n(lambda x: torch.sqrt(x), batch_size=128)(
            np.arange(1024).astype("float")
        ),
        np.arange(1024) ** 0.5,
    )


def test_t2n_examples():
    t2n(torch.zeros(10))
    t2n((torch.zeros(10, 2), torch.zeros(10)))


def test_numpy_dataset():
    ds = NumpyDataset({"a": np.arange(10), "b": 2 * np.arange(10)})
    assert len(ds) == 10
    assert ds[5] == {"a": 5, "b": 10}


def test_numpy_dataset_dtype():
    ds = NumpyDataset({"a": np.arange(10), "b": 2 * np.arange(10)}, dtype="float32")
    assert len(ds) == 10
    assert ds[3] == {"a": 3.0, "b": 6.0}


def test_numpy_dataset_nones():
    ds = NumpyDataset((np.arange(10), None), dtype=(("float32", "float32")))
    assert len(ds) == 10
    assert ds[7] == (7.0, None)


def test_numpy_dataset_filter():
    ds = NumpyDataset(np.arange(10), dtype="int64")
    assert len(ds) == 10

    ds = ds.filter(lambda ds: ds.data % 2 == 0)

    assert len(ds) == 5

    assert ds[0] == 0
    assert ds[1] == 2
    assert ds[2] == 4
    assert ds[3] == 6
    assert ds[4] == 8
