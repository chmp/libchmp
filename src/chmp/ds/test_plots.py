import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from chmp.ds import mpl_axis


def test_mpl_set_xscale():
    with mpl_axis(xscale="log") as ax:
        pass

    assert ax.get_xscale() == "log"


def test_mpl_set_yscale():
    with mpl_axis(yscale="log") as ax:
        pass

    assert ax.get_yscale() == "log"
