import operator as op

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from chmp.ds import (
    as_frame,
    filter_low_frequency_categories,
)


def test_filter_low_frequency_columns():
    actual = pd.DataFrame(
        {
            "a": pd.Series(["a"] * 5 + ["b"] * 4 + ["c"], dtype="category"),
            "b": pd.Series([1, 2, 3, 4, 5] * 2),
        }
    )

    actual = filter_low_frequency_categories(
        "a", min_frequency=0.2, other_category="other"
    ).fit_transform(actual)

    expected = pd.DataFrame(
        {
            "a": pd.Series(["a"] * 5 + ["b"] * 4 + ["other"], dtype="category"),
            "b": pd.Series([1, 2, 3, 4, 5] * 2),
        }
    )

    pdt.assert_frame_equal(actual, expected)
