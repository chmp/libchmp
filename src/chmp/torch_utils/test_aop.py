from chmp.torch_utils._aop import joinpoint, run_with_aspects, modify, add_aspect


def test_example():
    assert run_with_aspects(root, {}, 2) == 4

    assert run_with_aspects(root, {"inner": [aspect]}, 2) == -4


def test_modify():
    with modify(root) as my_root:
        add_aspect({"inner": [aspect]})

    assert root(2) == 4
    assert my_root(2) == -4


def aspect(proceed, val):
    return -proceed(val)


def root(val):
    return inner(val)


@joinpoint("inner")
def inner(val):
    return 2 * val
