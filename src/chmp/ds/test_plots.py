import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from chmp.ds import mpl_axis, mpl_figure


def test_mpl_set_xscale():
    with mpl_axis(xscale="log") as ax:
        pass

    assert ax.get_xscale() == "log"


def test_mpl_set_yscale():
    with mpl_axis(yscale="log") as ax:
        pass

    assert ax.get_yscale() == "log"


def test_mpl_figure():
    with mpl_figure() as _:
        pass

    with mpl_figure(n_axes=2) as (_, _):
        pass

    with mpl_figure(2, 2, n_axes=3) as (_, _, _):
        pass
