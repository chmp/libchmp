import pytest
import torch

from chmp.torch_utils import make_net


def test_make_net_ff():
    net = make_net(5).linear(10).relu().linear(10).relu().linear(2).build()

    assert net(torch.randn(10, 5)).shape == (10, 2)


def test_make_net_multiple_inputs():
    net = make_net(5, 3, 2).linear(10).relu().linear(10).relu().linear(2).build()

    res = net(
        torch.randn(10, 5),
        torch.randn(10, 3),
        torch.randn(10, 2),
    )

    assert res.shape == (10, 2)


@pytest.mark.parametrize(
    "activation", ["relu", "sigmoid", "tanh", "softplus", "squareplus"]
)
def test_activation_pipe(activation):
    net = make_net(5).linear(2).pipe(lambda n: getattr(n, activation)()).build()

    res = net(torch.randn(10, 5))
    assert res.shape == (10, 2)


@pytest.mark.parametrize(
    "activation", ["relu", "sigmoid", "tanh", "softplus", "squareplus"]
)
def test_activation_call(activation):
    net = make_net(5).linear(2).call(activation).build()

    res = net(torch.randn(10, 5))
    assert res.shape == (10, 2)
